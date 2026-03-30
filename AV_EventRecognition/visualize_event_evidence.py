from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class EvidenceCandidate:
    label: str
    score: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class LocalSoundRow:
    annotation_id: str
    participant_id: str
    video_id: str
    start: float
    stop: float
    description: str
    class_name: str
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an evidence-first AV event visualization with audio timeline, progressive evidence chips, "
            "object centers, pairwise distances, and the final recognized action."
        )
    )
    parser.add_argument("--report-json", type=Path, required=True, help="Per-clip AV pipeline report JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Defaults to *_evidence.mp4 next to the report.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.18,
        help="Mask overlay alpha used for the underlying visual detections.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Base thickness for connection lines.",
    )
    parser.add_argument(
        "--dot-radius",
        type=int,
        default=7,
        help="Base radius for detected center markers.",
    )
    return parser.parse_args()


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def parse_required_groups(raw_groups: list[list[str]]) -> list[list[str]]:
    return [[normalize_text(item) for item in group] for group in raw_groups]


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


def build_group_candidates(
    frame_dets: list[dict[str, Any]],
    required_groups: list[list[str]],
    required_group_min_scores: list[float],
) -> list[list[EvidenceCandidate]]:
    grouped: list[list[EvidenceCandidate]] = [[] for _ in required_groups]
    for det in frame_dets:
        center = center_from_detection(det)
        if center is None:
            continue
        label = normalize_text(det.get("label", ""))
        score = float(det.get("score", 0.0))
        for group_index, aliases in enumerate(required_groups):
            if score < required_group_min_scores[group_index]:
                continue
            if any(alias in label or label in alias for alias in aliases):
                grouped[group_index].append(
                    EvidenceCandidate(
                        label=label,
                        score=score,
                        center_x=center[0],
                        center_y=center[1],
                    )
                )
    return grouped


def select_best_combo(grouped: list[list[EvidenceCandidate]]) -> dict[str, Any] | None:
    if not grouped or any(not group for group in grouped):
        return None

    best_combo: dict[str, Any] | None = None
    for combo in itertools.product(*grouped):
        max_pairwise_distance = 0.0
        pairwise_distances: list[float] = []
        for left, right in itertools.combinations(combo, 2):
            distance = math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
            pairwise_distances.append(distance)
            max_pairwise_distance = max(max_pairwise_distance, distance)
        avg_score = sum(item.score for item in combo) / len(combo)
        current = {
            "max_pairwise_distance": max_pairwise_distance,
            "avg_score": avg_score,
            "detections": list(combo),
            "pairwise_distances": pairwise_distances,
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
    return best_combo


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_audio_from_video(video_path: Path, output_dir: Path) -> Path | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix="event_waveform_", suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass
        return None
    return wav_path


def mux_audio(silent_video: Path, output_path: Path, audio_source: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return silent_video
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode == 0 and output_path.exists():
        try:
            silent_video.unlink()
        except OSError:
            pass
        return output_path
    return silent_video


def create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ensure_parent(output_path)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create writer for {output_path}")
    return writer


def draw_chip(
    frame: np.ndarray,
    helper: Any,
    text: str,
    center_x: int,
    center_y: int,
    active: bool,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> tuple[int, int]:
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    width = text_width + 24
    height = text_height + baseline + 18
    x0 = center_x - width // 2
    y0 = center_y - height // 2
    x1 = x0 + width
    y1 = y0 + height
    fill_color = color if active else (70, 70, 70)
    helper.draw_rounded_rect(frame, x0, y0, x1, y1, max(8, height // 3), fill_color, alpha=0.85 if active else 0.52)
    text_color = (255, 255, 255) if active else (210, 210, 210)
    cv2.putText(
        frame,
        text,
        (x0 + 12, y1 - 10 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return center_x, center_y


def draw_evidence_panel(
    frame: np.ndarray,
    helper: Any,
    event_name: str,
    sound_label: str,
    required_groups: list[list[str]],
    threshold_px: float,
    frame_index: int,
    fps: float,
    first_seen_by_group: list[int | None],
    first_complete_frame: int | None,
    first_pass_frame: int | None,
    latest_combo: dict[str, Any] | None,
) -> None:
    h, w = frame.shape[:2]
    panel_h = max(150, int(h * 0.2))
    panel_y0 = 16
    panel_y1 = panel_y0 + panel_h
    panel_x0 = 18
    panel_x1 = w - 18
    helper.draw_rounded_rect(frame, panel_x0, panel_y0, panel_x1, panel_y1, 18, (18, 18, 18), alpha=0.72)

    title = f"Evidence Window: {event_name}"
    helper.draw_centered_text_line(
        frame=frame,
        y_baseline=panel_y0 + 28,
        parts=[(title, (245, 245, 245))],
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.75,
        thickness=2,
        pad_x=12,
        pad_y=8,
        gap_px=8,
        bg_alpha=0.0,
        radius=12,
    )

    sound_active = True
    step_labels = ["sound"] + [group[0] for group in required_groups] + ["geometry", "action"]
    step_colors = [(80, 180, 255)] + [helper.class_color_bgr(group[0]) for group in required_groups] + [(90, 220, 150), (40, 210, 90)]

    centers: list[tuple[int, int]] = []
    usable_width = panel_x1 - panel_x0 - 80
    start_x = panel_x0 + 40
    baseline_y = panel_y0 + 72
    for idx, label in enumerate(step_labels):
        if len(step_labels) == 1:
            cx = (panel_x0 + panel_x1) // 2
        else:
            cx = start_x + int(idx * (usable_width / (len(step_labels) - 1)))
        if idx == 0:
            active = sound_active
        elif 1 <= idx <= len(required_groups):
            seen_frame = first_seen_by_group[idx - 1]
            active = seen_frame is not None and frame_index >= seen_frame
        elif label == "geometry":
            active = first_complete_frame is not None and frame_index >= first_complete_frame
        else:
            active = first_pass_frame is not None and frame_index >= first_pass_frame
        centers.append(
            draw_chip(
                frame=frame,
                helper=helper,
                text=label,
                center_x=cx,
                center_y=baseline_y,
                active=active,
                color=step_colors[idx],
                font_scale=0.52,
                thickness=1,
            )
        )

    for left, right in zip(centers, centers[1:]):
        cv2.line(frame, left, right, (120, 120, 120), 2, cv2.LINE_AA)

    parts: list[tuple[str, tuple[int, int, int]]] = [(f"sound: {sound_label}", (80, 180, 255))]
    for group_index, group in enumerate(required_groups):
        seen_frame = first_seen_by_group[group_index]
        if seen_frame is None or frame_index < seen_frame:
            parts.append(("  > waiting", (170, 170, 170)))
            parts.append((f" {group[0]}", helper.class_color_bgr(group[0])))
        else:
            seen_time = seen_frame / max(1e-6, fps)
            parts.append(("  > ", (220, 220, 220)))
            parts.append((f"{group[0]}@{seen_time:.2f}s", helper.class_color_bgr(group[0])))

    if latest_combo is not None:
        status_color = (80, 230, 120) if latest_combo["passed"] else (70, 170, 255)
        parts.append(("  > max d=", (220, 220, 220)))
        comparison = "<" if latest_combo["max_pairwise_distance"] <= threshold_px else ">"
        parts.append((f"{latest_combo['max_pairwise_distance']:.1f}px {comparison} {threshold_px:.1f}px", status_color))
        if latest_combo["passed"] and first_pass_frame is not None and frame_index >= first_pass_frame:
            parts.append(("  > action ", (220, 220, 220)))
            parts.append((event_name, (80, 230, 120)))

    helper.draw_centered_text_line(
        frame=frame,
        y_baseline=panel_y0 + 124,
        parts=parts,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.53,
        thickness=1,
        pad_x=10,
        pad_y=8,
        gap_px=6,
        bg_alpha=0.0,
        radius=10,
    )


def draw_combo_overlay(
    frame: np.ndarray,
    helper: Any,
    combo: dict[str, Any] | None,
    required_groups: list[list[str]],
    threshold_px: float,
    dot_radius: int,
    line_thickness: int,
) -> None:
    if combo is None:
        return

    detections: list[EvidenceCandidate] = combo["detections"]
    points = [(int(round(det.center_x)), int(round(det.center_y))) for det in detections]
    passed = combo["max_pairwise_distance"] <= threshold_px
    line_color = (80, 220, 120) if passed else (70, 170, 255)

    for (idx_a, pt_a), (idx_b, pt_b) in itertools.combinations(enumerate(points), 2):
        cv2.line(frame, pt_a, pt_b, line_color, line_thickness, cv2.LINE_AA)
        midpoint = ((pt_a[0] + pt_b[0]) // 2, (pt_a[1] + pt_b[1]) // 2)
        distance = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
        cv2.putText(
            frame,
            f"{distance:.1f}",
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            line_color,
            1,
            cv2.LINE_AA,
        )

    for group, det, point in zip(required_groups, detections, points):
        color = helper.class_color_bgr(group[0])
        cv2.circle(frame, point, dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, point, dot_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{group[0]} {det.score:.2f}",
            (point[0] + 10, max(20, point[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    report_path = args.report_json.resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    repo_root = Path(__file__).resolve().parents[1]
    helper = load_module(repo_root / "ApplyYOLOE" / "predict_yoloe_seg_video.py", "predict_yoloe_seg_video")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = report["analysis"]
    predictions_path = Path(report["paths"]["predictions_json"])
    clip_path = Path(report["paths"]["clip_video"])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions JSON not found: {predictions_path}")
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip video not found: {clip_path}")

    output_path = args.output.resolve() if args.output else report_path.with_name(f"{report_path.stem}_evidence.mp4")
    silent_output_path = output_path.with_name(f"{output_path.stem}.silent{output_path.suffix}")

    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    frames: list[list[dict[str, Any]]] = payload.get("frames", [])
    required_groups = parse_required_groups(analysis["required_groups"])
    required_group_min_scores = analysis.get("required_group_min_scores", [0.0] * len(required_groups))
    threshold_px = float(analysis["distance_threshold_px"])

    first_seen_by_group: list[int | None] = [None] * len(required_groups)
    first_complete_frame: int | None = None
    first_pass_frame: int | None = None
    for frame_index, detections in enumerate(frames):
        grouped = build_group_candidates(detections, required_groups, required_group_min_scores)
        for group_index, candidates in enumerate(grouped):
            if candidates and first_seen_by_group[group_index] is None:
                first_seen_by_group[group_index] = frame_index
        combo = select_best_combo(grouped)
        if combo is not None and first_complete_frame is None:
            first_complete_frame = frame_index
        if combo is not None and combo["max_pairwise_distance"] <= threshold_px and first_pass_frame is None:
            first_pass_frame = frame_index

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open clip video {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_writer(silent_output_path, fps=fps, width=width, height=height)

    clip_duration = (len(frames) / fps) if fps > 0 else 0.0
    clip_audio_start = max(0.0, float(report["audio_hit"]["start_sec"]) - float(report["audio_hit"]["clip_start_sec"]))
    clip_audio_stop = max(clip_audio_start, float(report["audio_hit"]["end_sec"]) - float(report["audio_hit"]["clip_start_sec"]))
    local_sound_row = LocalSoundRow(
        annotation_id=str(report["audio_hit"]["row_index"]),
        participant_id=str(report["audio_hit"]["raw_row"].get("participant_id", "")),
        video_id=str(report["audio_hit"]["raw_row"].get("video_id", "")),
        start=clip_audio_start,
        stop=clip_audio_stop,
        description=str(report["audio_hit"]["label"]),
        class_name=str(report["audio_hit"]["label"]),
        class_id=int(report["audio_hit"]["raw_row"].get("class_id", 0) or 0),
    )

    temp_wav = extract_audio_from_video(clip_path, output_path.parent)
    waveform_envelope, waveform_hz, _wave_status = helper.load_waveform_envelope(temp_wav)
    timeline_canvas, timeline_pixels_per_second = helper.build_timeline_panel_canvas(
        rows=[local_sound_row],
        waveform_envelope=waveform_envelope,
        waveform_hz=waveform_hz,
        duration=clip_duration,
        panel_width=max(640, width - 80),
        panel_height=max(96, int(height * 0.14)),
        timeline_window_seconds=min(clip_duration if clip_duration > 0 else 15.0, 15.0),
    )
    text_style = helper.scaled_text_style(max(0.35, height / 720.0))

    persistent_combo: dict[str, Any] | None = None
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame_index >= len(frames):
                break

            detections = frames[frame_index]
            annotated = helper.render_cached_detections(frame, detections, alpha=args.mask_alpha)
            grouped = build_group_candidates(detections, required_groups, required_group_min_scores)
            combo = select_best_combo(grouped)
            if combo is not None:
                combo["passed"] = combo["max_pairwise_distance"] <= threshold_px
                persistent_combo = combo

            draw_combo_overlay(
                frame=annotated,
                helper=helper,
                combo=persistent_combo if all(item is not None for item in first_seen_by_group) and frame_index >= max(
                    seen or 0 for seen in first_seen_by_group
                ) else combo,
                required_groups=required_groups,
                threshold_px=threshold_px,
                dot_radius=args.dot_radius,
                line_thickness=args.line_thickness,
            )

            draw_evidence_panel(
                frame=annotated,
                helper=helper,
                event_name=report["predicted_event"] if analysis["matched"] else report["positive_event_name"] if "positive_event_name" in report else report["predicted_event"],
                sound_label=str(report["audio_hit"]["label"]),
                required_groups=required_groups,
                threshold_px=threshold_px,
                frame_index=frame_index,
                fps=fps,
                first_seen_by_group=first_seen_by_group,
                first_complete_frame=first_complete_frame,
                first_pass_frame=first_pass_frame,
                latest_combo=persistent_combo,
            )

            helper.render_audio_panel(
                frame=annotated,
                t=frame_index / max(1e-6, fps),
                duration=clip_duration,
                rows=[local_sound_row],
                timeline_canvas=timeline_canvas,
                timeline_pixels_per_second=timeline_pixels_per_second,
                waveform_envelope=waveform_envelope,
                waveform_hz=waveform_hz,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=max(0.35, 0.68 * max(0.35, height / 720.0)),
                thickness=max(1, int(round(2 * max(0.35, height / 720.0)))),
                text_style=text_style,
                timeline_window_seconds=min(clip_duration if clip_duration > 0 else 15.0, 15.0),
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if temp_wav is not None:
            try:
                temp_wav.unlink()
            except OSError:
                pass

    final_output = mux_audio(silent_output_path, output_path, clip_path)
    print(f"Saved evidence visualization to: {final_output}")


if __name__ == "__main__":
    main()
