from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass(frozen=True)
class AudioDetection:
    row_index: int
    raw: dict[str, str]
    start_sec: float
    end_sec: float
    label: str
    score: float | None


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    score: float
    center_x: float
    center_y: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run an audio-visual event pipeline: filter audio detections from CSV, extract clips with audio, "
            "run YOLOE prompted segmentation, then decide whether the full AV event is present."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file.")
    parser.add_argument("--audio-csv", type=Path, required=True, help="CSV generated from audio inference.")
    parser.add_argument(
        "--auto-generate-audio-csv",
        action="store_true",
        help="Run infer_hdf5_video.py to create --audio-csv before AV processing.",
    )
    parser.add_argument(
        "--refresh-audio-csv",
        action="store_true",
        help="Regenerate --audio-csv even if it already exists.",
    )
    parser.add_argument(
        "--audio-infer-script",
        type=Path,
        default=repo_root / "epic-sounds-annotations" / "src" / "tools" / "infer_hdf5_video.py",
        help="Path to EPIC audio inference script.",
    )
    parser.add_argument(
        "--audio-cfg",
        type=Path,
        default=repo_root / "epic-sounds-annotations" / "src" / "configs" / "EPIC-Sounds" / "ssast" / "SSAST_b_vit_p16.yaml",
        help="Config passed to infer_hdf5_video.py.",
    )
    parser.add_argument(
        "--audio-checkpoint",
        type=Path,
        default=repo_root / "SSAST_EPIC_SOUNDS.pyth",
        help="Checkpoint passed to infer_hdf5_video.py.",
    )
    parser.add_argument(
        "--audio-hdf5",
        type=Path,
        default=repo_root / "wav_to_pickled.hdf5",
        help="HDF5 file passed to infer_hdf5_video.py. Override if your exported bundle stores it elsewhere.",
    )
    parser.add_argument(
        "--audio-video-id",
        type=str,
        default="",
        help="Optional video-id for infer_hdf5_video.py. Defaults to the input video stem.",
    )
    parser.add_argument(
        "--audio-window-mode",
        type=str,
        default="annotation_regions",
        choices=["sliding", "peaks", "active_regions", "annotation_regions"],
        help="Window mode for infer_hdf5_video.py.",
    )
    parser.add_argument(
        "--audio-annotations-file",
        type=Path,
        default=repo_root / "epic-sounds-annotations" / "EPIC_Sounds_train.csv",
        help="Annotations file used with annotation_regions mode.",
    )
    parser.add_argument(
        "--audio-output-summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path for infer_hdf5_video.py. Defaults to --audio-csv with _summary suffix.",
    )
    parser.add_argument(
        "--recipe-json",
        type=Path,
        default=None,
        help="Optional JSON recipe file that fills audio label, event names, prompts, and must-have groups.",
    )
    parser.add_argument(
        "--audio-label",
        type=str,
        default="",
        help='Audio label to filter for, e.g. "chop". Matching is case-insensitive substring search.',
    )
    parser.add_argument(
        "--positive-event-name",
        type=str,
        default="",
        help="Name to emit when the AV evidence passes the required-object check.",
    )
    parser.add_argument(
        "--fallback-event-name",
        type=str,
        default="",
        help="Name to emit when the audio hit does not satisfy the AV constraints.",
    )
    parser.add_argument(
        "--must-have",
        action="append",
        default=[],
        help=(
            "Required visual class group. Repeat for each requirement. "
            'Use "|" to provide aliases, e.g. --must-have "cutting board|chopping board".'
        ),
    )
    parser.add_argument(
        "--must-have-min-score",
        action="append",
        default=[],
        help=(
            "Optional minimum score for each must-have group, in the same order as --must-have. "
            "Repeat once per group."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[],
        help="Visual prompts forwarded to the YOLOE video script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where extracted clips, YOLOE outputs, and reports will be written.",
    )
    parser.add_argument(
        "--yoloe-script",
        type=Path,
        default=repo_root / "ApplyYOLOE" / "predict_yoloe_seg_video.py",
        help="Path to the YOLOE segmentation video script.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fastsam-world",
        choices=["yoloe", "fastsam-world"],
        help="Backend to pass through to the YOLOE script.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "yoloe-11s-seg.pt"),
        help="YOLOE checkpoint path.",
    )
    parser.add_argument(
        "--world-model",
        type=str,
        default=str(repo_root / "yolov8m-world.pt"),
        help="YOLO-World checkpoint path for fastsam-world backend.",
    )
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default=str(repo_root / "FastSAM-s.pt"),
        help="FastSAM checkpoint path for fastsam-world backend.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help="Visual detection confidence threshold. Lower defaults help recover prompted objects in short EPIC clips.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help='Device override like "cpu" or "0".')
    parser.add_argument("--frame-stride", type=int, default=1, help="Only infer every Nth frame.")
    parser.add_argument("--resize-factor", type=float, default=1.0, help="Resize factor forwarded to YOLOE.")
    parser.add_argument("--mask-alpha", type=float, default=0.45, help="Mask alpha forwarded to YOLOE.")
    parser.add_argument(
        "--clip-pad-before",
        type=float,
        default=2.0,
        help="Extra context to include before each audio hit, in seconds.",
    )
    parser.add_argument(
        "--clip-pad-after",
        type=float,
        default=2.0,
        help="Extra context to include after each audio hit, in seconds.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Optional cap on the number of matching audio detections to process. 0 means all.",
    )
    parser.add_argument(
        "--distance-threshold-ratio",
        type=float,
        default=0.06,
        help="Max allowed pairwise center distance as a fraction of max(video_width, video_height).",
    )
    parser.add_argument(
        "--distance-threshold-px",
        type=float,
        default=-1.0,
        help="Optional absolute pixel threshold. Overrides --distance-threshold-ratio when > 0.",
    )
    parser.add_argument(
        "--min-support-frames",
        type=int,
        default=3,
        help="Minimum number of frames that must satisfy the object-closeness rule.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="avod_py171",
        help=(
            "Conda environment used to invoke the YOLOE script. "
            'Set to "" to disable conda and use --python-exe directly.'
        ),
    )
    parser.add_argument(
        "--conda-exe",
        type=Path,
        default=None,
        help="Optional explicit path to conda.exe. Defaults to the first conda on PATH.",
    )
    parser.add_argument(
        "--python-exe",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used when --conda-env is empty.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_recipe(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Recipe file must contain a JSON object: {path}")
    return payload


def merge_recipe_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.recipe_json is None:
        return args

    recipe_path = args.recipe_json.resolve()
    if not recipe_path.exists():
        raise FileNotFoundError(f"Recipe JSON not found: {recipe_path}")

    recipe = load_recipe(recipe_path)
    args.recipe_json = recipe_path

    if not args.audio_label:
        args.audio_label = str(recipe.get("audio_label", "")).strip()
    if not args.positive_event_name:
        args.positive_event_name = str(recipe.get("positive_event_name", "")).strip()
    if not args.fallback_event_name:
        args.fallback_event_name = str(recipe.get("fallback_event_name", "")).strip()
    if not args.prompts:
        args.prompts = [str(item) for item in recipe.get("prompts", [])]
    if not args.must_have:
        recipe_groups = recipe.get("must_have", [])
        merged_groups: list[str] = []
        for group in recipe_groups:
            if isinstance(group, str):
                merged_groups.append(group)
            elif isinstance(group, list):
                merged_groups.append("|".join(str(item) for item in group))
            else:
                raise ValueError(f"Invalid must_have group in recipe: {group!r}")
        args.must_have = merged_groups
    if not args.must_have_min_score:
        recipe_scores = recipe.get("must_have_min_scores", [])
        args.must_have_min_score = [str(item) for item in recipe_scores]

    if "distance_threshold_ratio" in recipe and args.distance_threshold_ratio == 0.06:
        args.distance_threshold_ratio = float(recipe["distance_threshold_ratio"])
    if "distance_threshold_px" in recipe and args.distance_threshold_px < 0:
        args.distance_threshold_px = float(recipe["distance_threshold_px"])
    if "min_support_frames" in recipe and args.min_support_frames == 3:
        args.min_support_frames = int(recipe["min_support_frames"])
    if "clip_pad_before" in recipe and args.clip_pad_before == 2.0:
        args.clip_pad_before = float(recipe["clip_pad_before"])
    if "clip_pad_after" in recipe and args.clip_pad_after == 2.0:
        args.clip_pad_after = float(recipe["clip_pad_after"])
    if "conf" in recipe and args.conf == 0.01:
        args.conf = float(recipe["conf"])

    if not args.audio_label:
        raise ValueError("Missing --audio-label and recipe did not provide audio_label.")
    if not args.positive_event_name:
        raise ValueError("Missing --positive-event-name and recipe did not provide positive_event_name.")
    if not args.fallback_event_name:
        args.fallback_event_name = f"other {args.audio_label} event"
    if not args.prompts:
        raise ValueError("Missing --prompts and recipe did not provide prompts.")
    if not args.must_have:
        raise ValueError("Missing --must-have and recipe did not provide must_have.")
    return args


def parse_timestamp(value: str) -> float:
    text = str(value).strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected timestamp format: {value}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def load_audio_hits(csv_path: Path, audio_label: str, max_clips: int) -> list[AudioDetection]:
    label_query = normalize_text(audio_label)
    hits: list[AudioDetection] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            label_fields = [
                row.get("class", ""),
                row.get("description", ""),
                row.get("top1_label", ""),
                row.get("source_class", ""),
                row.get("source_description", ""),
            ]
            haystack = " | ".join(normalize_text(value) for value in label_fields if value)
            if label_query not in haystack:
                continue

            start_value = row.get("start_sec")
            end_value = row.get("end_sec")
            if start_value not in (None, "") and end_value not in (None, ""):
                start_sec = float(start_value)
                end_sec = float(end_value)
            else:
                start_sec = parse_timestamp(row["start_timestamp"])
                end_sec = parse_timestamp(row["stop_timestamp"])

            score = None
            for key in ("top1_score", "score", "confidence"):
                if row.get(key) not in (None, ""):
                    score = float(row[key])
                    break

            hits.append(
                AudioDetection(
                    row_index=row_index,
                    raw=row,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    label=row.get("class") or row.get("description") or row.get("top1_label") or audio_label,
                    score=score,
                )
            )
            if max_clips > 0 and len(hits) >= max_clips:
                break
    return hits


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def parse_required_groups(values: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for value in values:
        aliases = [normalize_text(item) for item in value.split("|") if normalize_text(item)]
        if not aliases:
            raise ValueError(f"Invalid --must-have value: {value!r}")
        groups.append(aliases)
    return groups


def parse_required_group_min_scores(values: list[str], num_groups: int) -> list[float]:
    if not values:
        return [0.0] * num_groups
    if len(values) != num_groups:
        raise ValueError(
            f"Expected {num_groups} --must-have-min-score values to match --must-have groups, got {len(values)}."
        )
    return [max(0.0, float(value)) for value in values]


def ffprobe_video_duration(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    return float(result.stdout.strip())


def ffprobe_video_size(video_path: Path) -> tuple[int, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    width_text, height_text = result.stdout.strip().split("x", 1)
    return int(width_text), int(height_text)


def extract_subclip(input_video: Path, output_video: Path, start_sec: float, end_sec: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH.")
    ensure_parent(output_video)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-to",
        f"{max(0.0, end_sec):.3f}",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_video.exists():
        raise RuntimeError(f"ffmpeg clip extraction failed: {result.stderr.strip()}")


def build_yoloe_runner_command(args: argparse.Namespace) -> list[str]:
    conda_env = args.conda_env.strip()
    if conda_env:
        conda_exe = args.conda_exe.resolve() if args.conda_exe else None
        conda_cmd = str(conda_exe) if conda_exe is not None else shutil.which("conda")
        if not conda_cmd:
            raise RuntimeError(
                "Conda executable not found. Pass --conda-exe explicitly or set --conda-env \"\" to use --python-exe."
            )
        return [conda_cmd, "run", "--no-capture-output", "-n", conda_env, "python"]

    python_exe = args.python_exe.resolve()
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    return [str(python_exe)]


def build_python_runner_command(args: argparse.Namespace) -> list[str]:
    return build_yoloe_runner_command(args)


def find_conda_env_bin(conda_env: str) -> Path | None:
    conda_exe = shutil.which("conda")
    if not conda_exe:
        return None
    conda_root = Path(conda_exe).resolve().parents[1]
    env_path = conda_root / "envs" / conda_env
    candidate = env_path / "Library" / "bin"
    return candidate if candidate.exists() else None


def build_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    conda_env = args.conda_env.strip()
    if not conda_env:
        return env
    env_bin = find_conda_env_bin(conda_env)
    if env_bin is not None:
        current = env.get("PATH", "")
        env["PATH"] = str(env_bin) + os.pathsep + current
    return env


def default_audio_summary_csv(audio_csv_path: Path) -> Path:
    return audio_csv_path.with_name(f"{audio_csv_path.stem}_summary{audio_csv_path.suffix}")


def generate_audio_csv_if_needed(args: argparse.Namespace, audio_csv_path: Path, video_path: Path) -> None:
    should_generate = args.auto_generate_audio_csv or args.refresh_audio_csv or not audio_csv_path.exists()
    if not should_generate:
        return

    audio_infer_script = args.audio_infer_script.resolve()
    audio_cfg = args.audio_cfg.resolve()
    audio_checkpoint = args.audio_checkpoint.resolve()
    audio_hdf5 = args.audio_hdf5.resolve()
    audio_annotations_file = args.audio_annotations_file.resolve()
    audio_summary_csv = (
        args.audio_output_summary_csv.resolve() if args.audio_output_summary_csv is not None else default_audio_summary_csv(audio_csv_path)
    )
    audio_video_id = args.audio_video_id.strip() or video_path.stem
    audio_source_root = audio_infer_script.parent.parent

    for required_path, label in (
        (audio_infer_script, "audio infer script"),
        (audio_cfg, "audio cfg"),
        (audio_checkpoint, "audio checkpoint"),
        (audio_hdf5, "audio hdf5"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"{label} not found: {required_path}")
    if args.audio_window_mode == "annotation_regions" and not audio_annotations_file.exists():
        raise FileNotFoundError(f"audio annotations file not found: {audio_annotations_file}")

    ensure_parent(audio_csv_path)
    ensure_parent(audio_summary_csv)
    env = build_subprocess_env(args)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(audio_source_root) if not current_pythonpath else str(audio_source_root) + os.pathsep + current_pythonpath
    command: list[str] = [
        *build_python_runner_command(args),
        str(audio_infer_script),
        "--cfg",
        str(audio_cfg),
        "--checkpoint",
        str(audio_checkpoint),
        "--hdf5",
        str(audio_hdf5),
        "--video-id",
        audio_video_id,
        "--window-mode",
        args.audio_window_mode,
        "--output-csv",
        str(audio_csv_path),
        "--output-summary-csv",
        str(audio_summary_csv),
    ]
    if args.audio_window_mode == "annotation_regions":
        command.extend(["--annotations-file", str(audio_annotations_file)])

    print(f"Generating audio CSV for video_id={audio_video_id} -> {audio_csv_path}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(audio_source_root),
    )
    if result.returncode != 0 or not audio_csv_path.exists():
        raise RuntimeError(
            "Audio CSV generation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )


def run_yoloe_inference(args: argparse.Namespace, clip_path: Path, rendered_path: Path, predictions_path: Path) -> None:
    ensure_parent(rendered_path)
    ensure_parent(predictions_path)
    command: list[str] = [
        *build_yoloe_runner_command(args),
        str(args.yoloe_script),
        "--input",
        str(clip_path),
        "--output",
        str(rendered_path),
        "--backend",
        args.backend,
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--frame-stride",
        str(max(1, args.frame_stride)),
        "--resize-factor",
        str(args.resize_factor),
        "--mask-alpha",
        str(args.mask_alpha),
        "--copy-audio",
        "--save-predictions",
        str(predictions_path),
        "--prompts",
        *args.prompts,
    ]
    if args.backend == "yoloe":
        command.extend(["--model", args.model])
    else:
        command.extend(
            [
                "--world-model",
                args.world_model,
                "--fastsam-model",
                args.fastsam_model,
            ]
        )
    if args.device:
        command.extend(["--device", args.device])

    result = subprocess.run(command, capture_output=True, text=True, check=False, env=build_subprocess_env(args))
    if result.returncode != 0:
        raise RuntimeError(
            "YOLOE pipeline invocation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}"
        )


def center_from_detection(det: dict[str, Any]) -> tuple[float, float] | None:
    polygons = det.get("polygons") or []
    polygon_points: list[tuple[float, float]] = []
    for polygon in polygons:
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                polygon_points.append((float(point[0]), float(point[1])))
    if polygon_points:
        x = sum(point[0] for point in polygon_points) / len(polygon_points)
        y = sum(point[1] for point in polygon_points) / len(polygon_points)
        return x, y

    box = det.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = [float(value) for value in box]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def build_frame_candidates(
    frame_dets: list[dict[str, Any]],
    required_groups: list[list[str]],
    required_group_min_scores: list[float],
) -> list[list[DetectionCandidate]]:
    grouped: list[list[DetectionCandidate]] = [[] for _ in required_groups]
    for det in frame_dets:
        center = center_from_detection(det)
        if center is None:
            continue
        label = normalize_text(det.get("label", ""))
        score = float(det.get("score", 0.0))
        candidate = DetectionCandidate(
            label=label,
            score=score,
            center_x=center[0],
            center_y=center[1],
        )
        for group_index, aliases in enumerate(required_groups):
            if score < required_group_min_scores[group_index]:
                continue
            if any(alias in label or label in alias for alias in aliases):
                grouped[group_index].append(candidate)
    return grouped


def summarize_group_label(group_aliases: list[str]) -> str:
    return " | ".join(group_aliases)


def evaluate_frame(
    frame_index: int,
    frame_dets: list[dict[str, Any]],
    required_groups: list[list[str]],
    required_group_min_scores: list[float],
    distance_threshold_px: float,
) -> dict[str, Any]:
    grouped = build_frame_candidates(frame_dets, required_groups, required_group_min_scores)
    missing_groups = [summarize_group_label(required_groups[idx]) for idx, values in enumerate(grouped) if not values]
    if missing_groups:
        return {
            "frame_index": frame_index,
            "matched": False,
            "reason": "missing_required_classes",
            "missing_groups": missing_groups,
            "required_group_min_scores": required_group_min_scores,
        }

    best_combo: dict[str, Any] | None = None
    for combo in itertools.product(*grouped):
        max_pairwise_distance = 0.0
        for left, right in itertools.combinations(combo, 2):
            distance = math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
            max_pairwise_distance = max(max_pairwise_distance, distance)
        avg_score = sum(item.score for item in combo) / len(combo)
        current = {
            "max_pairwise_distance": max_pairwise_distance,
            "avg_score": avg_score,
            "detections": [
                {
                    "label": item.label,
                    "score": item.score,
                    "center": [item.center_x, item.center_y],
                }
                for item in combo
            ],
        }
        if best_combo is None:
            best_combo = current
            continue
        if current["max_pairwise_distance"] < best_combo["max_pairwise_distance"]:
            best_combo = current
            continue
        if (
            math.isclose(current["max_pairwise_distance"], best_combo["max_pairwise_distance"])
            and current["avg_score"] > best_combo["avg_score"]
        ):
            best_combo = current

    assert best_combo is not None
    matched = best_combo["max_pairwise_distance"] <= distance_threshold_px
    best_combo.update(
        {
            "frame_index": frame_index,
            "matched": matched,
            "distance_threshold_px": distance_threshold_px,
            "reason": "passed" if matched else "objects_too_far_apart",
        }
    )
    return best_combo


def analyze_predictions(
    predictions_path: Path,
    required_groups: list[list[str]],
    required_group_min_scores: list[float],
    clip_video_path: Path,
    distance_threshold_px_override: float,
    distance_threshold_ratio: float,
    min_support_frames: int,
) -> dict[str, Any]:
    with predictions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frames = payload.get("frames", [])

    width, height = ffprobe_video_size(clip_video_path)
    if distance_threshold_px_override > 0:
        threshold_px = float(distance_threshold_px_override)
    else:
        threshold_px = max(width, height) * float(distance_threshold_ratio)

    frame_results = [
        evaluate_frame(
            frame_index=frame_index,
            frame_dets=frame_dets,
            required_groups=required_groups,
            required_group_min_scores=required_group_min_scores,
            distance_threshold_px=threshold_px,
        )
        for frame_index, frame_dets in enumerate(frames)
    ]

    support_frames = [result for result in frame_results if result.get("matched")]
    best_frame = None
    if frame_results:
        best_frame = min(
            frame_results,
            key=lambda item: (
                0 if item.get("matched") else 1,
                item.get("max_pairwise_distance", float("inf")),
                -item.get("avg_score", 0.0),
            ),
        )

    return {
        "num_frames": len(frames),
        "video_size": {"width": width, "height": height},
        "required_groups": required_groups,
        "required_group_min_scores": required_group_min_scores,
        "distance_threshold_px": threshold_px,
        "distance_threshold_ratio": distance_threshold_ratio,
        "min_support_frames": min_support_frames,
        "num_support_frames": len(support_frames),
        "matched": len(support_frames) >= max(1, min_support_frames),
        "best_frame": best_frame,
        "support_frame_indices": [item["frame_index"] for item in support_frames],
        "frame_results": frame_results,
    }


def make_clip_stem(audio_hit: AudioDetection) -> str:
    return f"hit_{audio_hit.row_index:05d}_{audio_hit.start_sec:08.3f}_{audio_hit.end_sec:08.3f}".replace(".", "_")


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = merge_recipe_args(parse_args())
    video_path = args.video.resolve()
    audio_csv_path = args.audio_csv.resolve()
    output_dir = args.output_dir.resolve()
    yoloe_script = args.yoloe_script.resolve()
    subprocess_env = build_subprocess_env(args)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video type: {video_path}")
    if not yoloe_script.exists():
        raise FileNotFoundError(f"YOLOE script not found: {yoloe_script}")

    generate_audio_csv_if_needed(args, audio_csv_path=audio_csv_path, video_path=video_path)
    if not audio_csv_path.exists():
        raise FileNotFoundError(f"Audio CSV not found: {audio_csv_path}")

    required_groups = parse_required_groups(args.must_have)
    required_group_min_scores = parse_required_group_min_scores(args.must_have_min_score, len(required_groups))
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    renders_dir = output_dir / "rendered"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for path in (clips_dir, renders_dir, predictions_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    audio_hits = load_audio_hits(audio_csv_path, args.audio_label, args.max_clips)
    if not audio_hits:
        raise RuntimeError(
            f'No rows in {audio_csv_path} matched audio label "{args.audio_label}".'
        )

    original_path = os.environ.get("PATH", "")
    summary_records: list[dict[str, Any]] = []
    try:
        os.environ["PATH"] = subprocess_env.get("PATH", original_path)
        video_duration = ffprobe_video_duration(video_path)

        for hit_index, audio_hit in enumerate(audio_hits):
            clip_start = max(0.0, audio_hit.start_sec - args.clip_pad_before)
            clip_end = min(video_duration, audio_hit.end_sec + args.clip_pad_after)
            if clip_end <= clip_start:
                clip_end = min(video_duration, clip_start + 0.5)

            clip_stem = make_clip_stem(audio_hit)
            clip_path = clips_dir / f"{clip_stem}.mp4"
            rendered_path = renders_dir / f"{clip_stem}_vis.mp4"
            predictions_path = predictions_dir / f"{clip_stem}.json"
            report_path = reports_dir / f"{clip_stem}.json"

            print(
                f"[{hit_index + 1}/{len(audio_hits)}] Processing audio hit "
                f"{audio_hit.label!r} from {clip_start:.3f}s to {clip_end:.3f}s"
            )

            extract_subclip(video_path, clip_path, clip_start, clip_end)
            run_yoloe_inference(args, clip_path, rendered_path, predictions_path)
            analysis = analyze_predictions(
                predictions_path=predictions_path,
                required_groups=required_groups,
                required_group_min_scores=required_group_min_scores,
                clip_video_path=clip_path,
                distance_threshold_px_override=args.distance_threshold_px,
                distance_threshold_ratio=args.distance_threshold_ratio,
                min_support_frames=max(1, args.min_support_frames),
            )
            predicted_event = args.positive_event_name if analysis["matched"] else args.fallback_event_name

            record = {
                "clip_id": clip_stem,
                "predicted_event": predicted_event,
                "audio_hit": {
                    "row_index": audio_hit.row_index,
                    "label": audio_hit.label,
                    "score": audio_hit.score,
                    "start_sec": audio_hit.start_sec,
                    "end_sec": audio_hit.end_sec,
                    "clip_start_sec": clip_start,
                    "clip_end_sec": clip_end,
                    "raw_row": audio_hit.raw,
                },
                "visual_prompts": args.prompts,
                "paths": {
                    "clip_video": str(clip_path),
                    "rendered_video": str(rendered_path),
                    "predictions_json": str(predictions_path),
                },
                "analysis": analysis,
            }
            write_json(report_path, record)
            summary_records.append(
                {
                    "clip_id": clip_stem,
                    "predicted_event": predicted_event,
                    "audio_label": audio_hit.label,
                    "audio_start_sec": audio_hit.start_sec,
                    "audio_end_sec": audio_hit.end_sec,
                    "clip_start_sec": clip_start,
                    "clip_end_sec": clip_end,
                    "matched": analysis["matched"],
                    "num_support_frames": analysis["num_support_frames"],
                    "best_frame_index": None if analysis["best_frame"] is None else analysis["best_frame"]["frame_index"],
                    "best_frame_max_pairwise_distance": (
                        None if analysis["best_frame"] is None else analysis["best_frame"].get("max_pairwise_distance")
                    ),
                    "report_json": str(report_path),
                }
            )
    finally:
        os.environ["PATH"] = original_path

    summary_payload = {
        "video": str(video_path),
        "audio_csv": str(audio_csv_path),
        "recipe_json": None if args.recipe_json is None else str(args.recipe_json),
        "audio_label": args.audio_label,
        "positive_event_name": args.positive_event_name,
        "fallback_event_name": args.fallback_event_name,
        "required_groups": required_groups,
        "required_group_min_scores": required_group_min_scores,
        "prompts": args.prompts,
        "num_audio_hits": len(audio_hits),
        "results": summary_records,
    }
    write_json(output_dir / "summary.json", summary_payload)
    print(f"Saved AV event summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
