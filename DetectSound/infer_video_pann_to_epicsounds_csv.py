from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from label_mapping import load_label_to_id, map_audioset_label_to_epic


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIOSET_REPO = REPO_ROOT / "audioset_tagging_cnn"
PYTORCH_DIR = AUDIOSET_REPO / "pytorch"
METADATA_CSV = AUDIOSET_REPO / "metadata" / "class_labels_indices.csv"
EPIC_METADATA_CSV = REPO_ROOT / "epic-sounds-annotations" / "class_labels_indices.csv"


@dataclass(frozen=True)
class DetectionEvent:
    label: str
    class_id: int
    start_seconds: float
    stop_seconds: float
    score: float


@dataclass(frozen=True)
class CsvEventRow:
    start_seconds: float
    stop_seconds: float
    output_label: str
    output_class_id: int
    audioset_label: str
    audioset_class_id: int
    epic_label: str
    epic_class_id: str
    mapping_status: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract audio from a video, run PANN sound event detection, and export "
            "an EPIC Sounds-style CSV."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the input video file.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to the output CSV file.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to a PANN checkpoint, e.g. Cnn14_DecisionLevelMax_mAP=0.385.pth.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="Cnn14_DecisionLevelMax",
        help="PANN model class name. Use a DecisionLevel model for event timestamps.",
    )
    parser.add_argument(
        "--extracted-audio",
        type=Path,
        default=None,
        help="Optional path to save the extracted WAV. Defaults to a temporary file.",
    )
    parser.add_argument("--sample-rate", type=int, default=32000, help="Audio sample rate for PANN inference.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--mel-bins", type=int, default=64)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=14000)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Framewise probability threshold used to form event segments.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.20,
        help="Minimum event duration in seconds after thresholding.",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.12,
        help="Merge neighboring segments of the same class when the gap is at most this many seconds.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=15,
        help="Only decode the top-N labels ranked by peak framewise score.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=250,
        help="Maximum number of events to write after sorting by time and confidence.",
    )
    parser.add_argument(
        "--csv-sample-rate",
        type=int,
        default=24000,
        help="Sample rate used for start_sample and stop_sample in the output CSV.",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default=None,
        help="Optional participant_id override. Defaults to the prefix before the first underscore in the video id.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Optional video_id override. Defaults to the input filename stem.",
    )
    parser.add_argument(
        "--include-score-column",
        action="store_true",
        help="Append a score column after class_id for debugging/filter tuning.",
    )
    parser.add_argument(
        "--label-space",
        type=str,
        choices=["audioset", "epicsounds"],
        default="audioset",
        help="Write raw AudioSet labels or map them to EPIC-Sounds labels when possible.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Run inference on CUDA when available.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_ffprobe_key_value_output(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def probe_audio_duration_seconds(audio_path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found on PATH")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}: {result.stderr.strip()}")

    meta = parse_ffprobe_key_value_output(result.stdout)
    duration_value = meta.get("duration", "0")
    return float(duration_value) if duration_value not in {"", "N/A"} else 0.0


def probe_video_duration_seconds(video_path: Path) -> float:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found on PATH")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")

    meta = parse_ffprobe_key_value_output(result.stdout)
    duration_value = meta.get("duration", "0")
    return float(duration_value) if duration_value not in {"", "N/A"} else 0.0


def extract_audio_from_video(video_path: Path, output_audio_path: Path, sample_rate: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found on PATH")

    ensure_parent_dir(output_audio_path)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(output_audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not output_audio_path.exists():
        raise RuntimeError(result.stderr.strip() or f"ffmpeg audio extraction failed for {video_path}")


def load_labels(metadata_csv: Path) -> list[str]:
    with metadata_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["display_name"] for row in reader]


def load_model(args: argparse.Namespace, classes_num: int) -> torch.nn.Module:
    if str(PYTORCH_DIR) not in sys.path:
        sys.path.insert(0, str(PYTORCH_DIR))

    import models as pann_models

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model_cls = getattr(pann_models, args.model_type, None)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model_cls(
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size,
        mel_bins=args.mel_bins,
        fmin=args.fmin,
        fmax=args.fmax,
        classes_num=classes_num,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def infer_framewise_output(
    audio_path: Path,
    model: torch.nn.Module,
    sample_rate: int,
) -> np.ndarray:
    if str(PYTORCH_DIR) not in sys.path:
        sys.path.insert(0, str(PYTORCH_DIR))

    from pytorch_utils import move_data_to_device

    waveform = load_mono_wav(audio_path, sample_rate)[None, :]
    device = next(model.parameters()).device
    waveform_tensor = move_data_to_device(waveform, device)

    with torch.no_grad():
        output_dict = model(waveform_tensor, None)

    if "framewise_output" not in output_dict:
        raise RuntimeError(
            f"Model {model.__class__.__name__} does not expose framewise_output. "
            "Use a DecisionLevel model for EPIC Sounds-style intervals."
        )

    return output_dict["framewise_output"].detach().cpu().numpy()[0]


def load_mono_wav(audio_path: Path, expected_sample_rate: int) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw_bytes = wav_file.readframes(frame_count)

    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"Unexpected WAV sample rate {sample_rate} for {audio_path}; expected {expected_sample_rate}"
        )
    if channels != 1:
        raise ValueError(f"Expected mono WAV but found {channels} channels in {audio_path}")

    if sample_width == 1:
        waveform = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        waveform = (waveform - 128.0) / 128.0
    elif sample_width == 2:
        waveform = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        waveform = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} bytes in {audio_path}")

    return np.ascontiguousarray(waveform, dtype=np.float32)


def merge_segments(
    segments: list[tuple[int, int, float]],
    frames_per_second: float,
    merge_gap_seconds: float,
) -> list[tuple[int, int, float]]:
    if not segments:
        return []

    merged: list[list[float]] = [[segments[0][0], segments[0][1], segments[0][2]]]
    max_gap_frames = int(round(merge_gap_seconds * frames_per_second))

    for start_idx, stop_idx, score in segments[1:]:
        last = merged[-1]
        if start_idx - int(last[1]) <= max_gap_frames + 1:
            last[1] = max(last[1], stop_idx)
            last[2] = max(last[2], score)
        else:
            merged.append([start_idx, stop_idx, score])

    return [(int(start), int(stop), float(score)) for start, stop, score in merged]


def decode_events(
    framewise_output: np.ndarray,
    labels: list[str],
    threshold: float,
    min_duration: float,
    merge_gap: float,
    top_labels: int,
    sample_rate: int,
    hop_size: int,
    max_events: int,
) -> list[DetectionEvent]:
    frames_per_second = sample_rate / hop_size
    peak_scores = np.max(framewise_output, axis=0)
    ranked_label_indexes = np.argsort(peak_scores)[::-1][: min(top_labels, len(labels))]

    events: list[DetectionEvent] = []
    min_frames = max(1, int(round(min_duration * frames_per_second)))

    for class_idx in ranked_label_indexes:
        scores = framewise_output[:, class_idx]
        above_threshold = scores >= threshold
        segments: list[tuple[int, int, float]] = []
        start_idx: int | None = None

        for frame_idx, active in enumerate(above_threshold):
            if active and start_idx is None:
                start_idx = frame_idx
            elif not active and start_idx is not None:
                stop_idx = frame_idx - 1
                if stop_idx - start_idx + 1 >= min_frames:
                    segment_score = float(np.max(scores[start_idx : stop_idx + 1]))
                    segments.append((start_idx, stop_idx, segment_score))
                start_idx = None

        if start_idx is not None:
            stop_idx = len(scores) - 1
            if stop_idx - start_idx + 1 >= min_frames:
                segment_score = float(np.max(scores[start_idx : stop_idx + 1]))
                segments.append((start_idx, stop_idx, segment_score))

        for start_idx, stop_idx, score in merge_segments(segments, frames_per_second, merge_gap):
            start_seconds = start_idx / frames_per_second
            stop_seconds = (stop_idx + 1) / frames_per_second
            events.append(
                DetectionEvent(
                    label=labels[class_idx],
                    class_id=int(class_idx),
                    start_seconds=float(start_seconds),
                    stop_seconds=float(stop_seconds),
                    score=float(score),
                )
            )

    events.sort(key=lambda event: (event.start_seconds, -(event.score), event.class_id))
    return events[:max_events]


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole_seconds = int(seconds)
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def seconds_to_sample_index(seconds: float, sample_rate: int) -> int:
    return int(round(max(0.0, seconds) * sample_rate))


def write_events_csv(
    output_csv: Path,
    video_id: str,
    participant_id: str,
    rows: list[CsvEventRow],
    csv_sample_rate: int,
    include_score_column: bool,
) -> None:
    ensure_parent_dir(output_csv)
    fieldnames = [
        "annotation_id",
        "participant_id",
        "video_id",
        "start_timestamp",
        "stop_timestamp",
        "start_sample",
        "stop_sample",
        "description",
        "class",
        "class_id",
        "audioset_label",
        "audioset_class_id",
        "epic_label",
        "epic_class_id",
        "mapping_status",
    ]
    if include_score_column:
        fieldnames.append("score")

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, row_data in enumerate(rows):
            row = {
                "annotation_id": f"{video_id}_{index}",
                "participant_id": participant_id,
                "video_id": video_id,
                "start_timestamp": format_timestamp(row_data.start_seconds),
                "stop_timestamp": format_timestamp(row_data.stop_seconds),
                "start_sample": seconds_to_sample_index(row_data.start_seconds, csv_sample_rate),
                "stop_sample": seconds_to_sample_index(row_data.stop_seconds, csv_sample_rate),
                "description": row_data.output_label,
                "class": row_data.output_label,
                "class_id": row_data.output_class_id,
                "audioset_label": row_data.audioset_label,
                "audioset_class_id": row_data.audioset_class_id,
                "epic_label": row_data.epic_label,
                "epic_class_id": row_data.epic_class_id,
                "mapping_status": row_data.mapping_status,
            }
            if include_score_column:
                row["score"] = f"{row_data.score:.6f}"
            writer.writerow(row)


def build_csv_rows(
    events: list[DetectionEvent],
    label_space: str,
    epic_metadata_csv: Path,
) -> list[CsvEventRow]:
    epic_label_to_id = load_label_to_id(epic_metadata_csv)
    rows: list[CsvEventRow] = []

    for event in events:
        mapped = map_audioset_label_to_epic(event.label, epic_label_to_id)
        if mapped is None:
            epic_label = ""
            epic_id = ""
            mapping_status = "unmapped"
            output_label = event.label
            output_class_id = event.class_id
        else:
            epic_label, epic_id_int = mapped
            epic_id = str(epic_id_int)
            mapping_status = "mapped"
            if label_space == "epicsounds":
                output_label = epic_label
                output_class_id = epic_id_int
            else:
                output_label = event.label
                output_class_id = event.class_id

        rows.append(
            CsvEventRow(
                start_seconds=event.start_seconds,
                stop_seconds=event.stop_seconds,
                output_label=output_label,
                output_class_id=output_class_id,
                audioset_label=event.label,
                audioset_class_id=event.class_id,
                epic_label=epic_label,
                epic_class_id=epic_id,
                mapping_status=mapping_status,
                score=event.score,
            )
        )

    rows.sort(key=lambda row: (row.start_seconds, -(row.score), row.output_class_id))
    return rows


def derive_participant_id(video_id: str, override: str | None) -> str:
    if override:
        return override
    return video_id.split("_", 1)[0]


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    if args.label_space == "epicsounds" and not EPIC_METADATA_CSV.exists():
        raise FileNotFoundError(f"EPIC metadata CSV not found: {EPIC_METADATA_CSV}")

    video_id = args.video_id or args.input.stem
    participant_id = derive_participant_id(video_id, args.participant_id)
    labels = load_labels(METADATA_CSV)

    keep_audio = args.extracted_audio is not None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if keep_audio:
        audio_path = args.extracted_audio
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="pann_audio_", dir=str(REPO_ROOT))
        audio_path = Path(temp_dir.name) / f"{video_id}.wav"

    try:
        extract_audio_from_video(args.input, audio_path, args.sample_rate)
        model = load_model(args, len(labels))
        framewise_output = infer_framewise_output(audio_path, model, args.sample_rate)
        events = decode_events(
            framewise_output=framewise_output,
            labels=labels,
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            top_labels=args.top_labels,
            sample_rate=args.sample_rate,
            hop_size=args.hop_size,
            max_events=args.max_events,
        )

        audio_duration = probe_audio_duration_seconds(audio_path)
        video_duration = probe_video_duration_seconds(args.input)
        capped_events: list[DetectionEvent] = []
        for event in events:
            start_seconds = min(event.start_seconds, video_duration, audio_duration)
            stop_seconds = min(event.stop_seconds, video_duration, audio_duration)
            if stop_seconds <= start_seconds:
                continue
            capped_events.append(
                DetectionEvent(
                    label=event.label,
                    class_id=event.class_id,
                    start_seconds=start_seconds,
                    stop_seconds=stop_seconds,
                    score=event.score,
                )
            )

        csv_rows = build_csv_rows(
            events=capped_events,
            label_space=args.label_space,
            epic_metadata_csv=EPIC_METADATA_CSV,
        )

        write_events_csv(
            output_csv=args.output_csv,
            video_id=video_id,
            participant_id=participant_id,
            rows=csv_rows,
            csv_sample_rate=args.csv_sample_rate,
            include_score_column=args.include_score_column,
        )

        print(f"Extracted audio: {audio_path}")
        print(f"Detected events: {len(csv_rows)}")
        print(f"Output CSV: {args.output_csv}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
