from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

# Some videos contain multiple streams and can trigger OpenCV's FFmpeg retry
# limit when decoding. Raise the limit before importing cv2.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "65536")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SoundRow:
    annotation_id: str
    participant_id: str
    video_id: str
    start: float
    stop: float
    description: str
    class_name: str
    class_id: int


# Define the CLI for prompted segmentation, rendering, caching, and optional audio overlays.
def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run prompted segmentation on a video or a directory of frames and save the rendered detections "
            "as a video."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an input video file or a directory containing image frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the rendered output video, e.g. output.mp4.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="yoloe",
        choices=["yoloe", "fastsam-world"],
        help="Backend to use: YOLOE prompted segmentation or YOLO-World detections refined by FastSAM masks.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo_root / "yoloe-11s-seg.pt"),
        help="YOLOE segmentation checkpoint to load when --backend yoloe is used.",
    )
    parser.add_argument(
        "--world-model",
        type=str,
        default=str(repo_root / "yolov8m-world.pt"),
        help="YOLO-World checkpoint to load when --backend fastsam-world is used.",
    )
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default=str(repo_root / "FastSAM-s.pt"),
        help="FastSAM checkpoint to load when --backend fastsam-world is used.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["hands", "door"],
        help='Text prompts to use. Defaults to: "hands door".',
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override like "cpu", "0", or "0,1".',
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS to use when the input is a frames directory.",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=1.0,
        help="Scale decoded/rendered frames by this factor to speed up processing, e.g. 0.5.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run inference every Nth frame and reuse the last detections for skipped frames.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Mask overlay opacity for the FastSAM + YOLO-World backend.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "Optional cap on processed duration. Useful for generating a short preview "
            "video first to verify prompts and rendering before running the full video."
        ),
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save each rendered frame next to the output video under a *_frames directory.",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="When the input is a video file, mux the source audio into the final output video.",
    )
    parser.add_argument(
        "--visualize-audio",
        action="store_true",
        help="Overlay an audio waveform timeline panel on the output video.",
    )
    parser.add_argument(
        "--audio-source",
        type=Path,
        default=None,
        help="Optional WAV/audio file to use for waveform visualization. Defaults to the input video's audio track.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="",
        help="Optional EPIC video_id override used when loading CSV sound labels.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=repo_root / "epic-sounds-annotations" / "EPIC_Sounds_train.csv",
        help="Path to EPIC_Sounds_train.csv.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=repo_root / "epic-sounds-annotations" / "EPIC_Sounds_validation.csv",
        help="Path to EPIC_Sounds_validation.csv.",
    )
    parser.add_argument(
        "--csv-split",
        type=str,
        choices=["train", "validation", "auto"],
        default="auto",
        help="Which EPIC CSV split to use for sound labels.",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional JSON path to save per-frame detections for later re-rendering without inference.",
    )
    parser.add_argument(
        "--load-predictions",
        type=Path,
        default=None,
        help="Optional JSON path of saved detections to render instead of running model inference.",
    )
    return parser.parse_args()


# Collect and sort image frames when the input is a directory instead of a video file.
def collect_frame_paths(frames_dir: Path) -> list[Path]:
    frame_paths = sorted(
        path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return frame_paths


# Ensure the output directory for a file path exists before writing.
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Parse EPIC-style HH:MM:SS.sss timestamps into seconds.
def parse_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected timestamp format: {ts}")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


# Parse ffprobe's key=value text output into a dictionary.
def parse_ffprobe_key_value_output(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


# Probe a video with ffprobe to get fps, frame count, frame size, and duration.
def probe_video_with_ffmpeg(video_path: Path) -> tuple[float, int, int, int, float]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe not found on PATH")

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")

    meta = parse_ffprobe_key_value_output(result.stdout)
    width = int(meta["width"])
    height = int(meta["height"])

    fps_value = meta.get("avg_frame_rate") or meta.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(fps_value))
    except Exception as exc:
        raise RuntimeError(f"Could not parse fps '{fps_value}' for {video_path}") from exc
    if fps <= 0:
        raise RuntimeError(f"Invalid fps from ffprobe for {video_path}: {fps_value}")

    duration_value = meta.get("duration", "0")
    duration = float(duration_value) if duration_value not in {"N/A", ""} else 0.0

    nb_frames_value = meta.get("nb_frames", "0")
    if nb_frames_value not in {"N/A", ""}:
        frame_count = int(nb_frames_value)
    elif duration > 0:
        frame_count = int(round(duration * fps))
    else:
        frame_count = 0

    if duration <= 0 and frame_count > 0:
        duration = frame_count / fps

    return fps, frame_count, width, height, duration


class FfmpegVideoReader:
    # Stream decoded raw BGR frames from ffmpeg to avoid flaky OpenCV video decoding paths.
    def __init__(self, video_path: Path, width: int, height: int) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg not found on PATH")

        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3
        command = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vsync",
            "0",
            "-",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read one raw frame from the ffmpeg pipe and reshape it into an HxWx3 image.
    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.process.stdout is None:
            return False, None

        raw = self.process.stdout.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            return False, None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    # Close the ffmpeg process and its pipes cleanly.
    def release(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


# Add the original audio track back onto a rendered silent video.
def mux_audio_into_video(
    silent_video_path: Path,
    output_path: Path,
    audio_source: Path,
    duration_seconds: float,
) -> tuple[bool, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "ffmpeg not found on PATH"

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video_path),
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
        "-b:a",
        "192k",
        "-t",
        f"{max(0.0, duration_seconds):.3f}",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and output_path.exists():
        return True, f"muxed audio from {audio_source}"
    return False, result.stderr.strip() or "ffmpeg audio mux failed"


# Create the OpenCV video writer used for saving the rendered output.
def create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    ensure_parent_dir(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")
    return writer


# Apply the optional resize factor and keep the output dimensions codec-friendly.
def compute_output_size(width: int, height: int, resize_factor: float) -> tuple[int, int]:
    if resize_factor <= 0:
        raise ValueError("--resize-factor must be > 0")
    out_width = max(1, int(round(width * resize_factor)))
    out_height = max(1, int(round(height * resize_factor)))
    out_width += out_width % 2
    out_height += out_height % 2
    return out_width, out_height


# Draw a rounded rectangle with optional alpha blending for UI panels and text backgrounds.
def draw_rounded_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 + radius, y0), (x1 - radius, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x0, y0 + radius), (x1, y1 - radius), color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x0 + radius, y0 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 - radius, y0 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x0 + radius, y1 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (x1 - radius, y1 - radius), radius, color, -1, cv2.LINE_AA)

    if alpha >= 0.999:
        frame[:, :] = overlay
    else:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# Draw a centered line of colored text parts with a rounded translucent background.
def draw_centered_text_line(
    frame: np.ndarray,
    y_baseline: int,
    parts: list[tuple[str, tuple[int, int, int]]],
    font: int,
    font_scale: float,
    thickness: int,
    pad_x: int = 14,
    pad_y: int = 10,
    gap_px: int = 8,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.58,
    radius: int = 14,
) -> None:
    h, w = frame.shape[:2]
    if not parts:
        return

    sizes: list[tuple[int, int]] = []
    total_w = 0
    max_h = 0
    for text, _ in parts:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        sizes.append((tw, th))
        total_w += tw
        max_h = max(max_h, th)
    total_w += gap_px * (len(parts) - 1)

    x0 = max(10, int((w - total_w) / 2))
    x1 = min(w - 10, x0 + total_w)
    rect_x0 = max(8, x0 - pad_x)
    rect_x1 = min(w - 8, x1 + pad_x)
    rect_y0 = max(8, y_baseline - max_h - pad_y)
    rect_y1 = min(h - 8, y_baseline + pad_y)

    draw_rounded_rect(frame, rect_x0, rect_y0, rect_x1, rect_y1, radius, bg_color, alpha=bg_alpha)

    x = x0
    for (text, color), (tw, _th) in zip(parts, sizes):
        cv2.putText(frame, text, (x, y_baseline), font, font_scale, color, thickness, cv2.LINE_AA)
        x += tw + gap_px


# Scale a UI size value while enforcing a readable minimum.
def scaled_ui_value(base_value: int | float, resize_factor: float, minimum: int) -> int:
    return max(minimum, int(round(base_value * resize_factor)))


# Build a small set of scaled text and panel spacing values for overlays.
def scaled_text_style(resize_factor: float) -> dict[str, int]:
    ui_scale = max(0.35, resize_factor)
    return {
        "pad_x": scaled_ui_value(14, ui_scale, 6),
        "pad_y": scaled_ui_value(10, ui_scale, 4),
        "gap_px": scaled_ui_value(8, ui_scale, 3),
        "radius": scaled_ui_value(14, ui_scale, 5),
    }


# Assign a deterministic display color to each class label.
def class_color_bgr(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# Reuse the same color mapping helper when only a numeric class id is available.
def class_id_to_color_bgr(class_id: int) -> tuple[int, int, int]:
    return class_color_bgr(str(class_id))


# Load EPIC sound annotation rows for one video so they can be shown on the audio timeline overlay.
def load_sound_rows(csv_path: Path, video_id: str) -> list[SoundRow]:
    if not csv_path.exists():
        return []

    rows: list[SoundRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("video_id") != video_id:
                continue
            rows.append(
                SoundRow(
                    annotation_id=r.get("annotation_id", ""),
                    participant_id=r.get("participant_id", ""),
                    video_id=r.get("video_id", ""),
                    start=parse_timestamp(r["start_timestamp"]),
                    stop=parse_timestamp(r["stop_timestamp"]),
                    description=r.get("description", ""),
                    class_name=r.get("class", ""),
                    class_id=int(r["class_id"]),
                )
            )
    rows.sort(key=lambda item: item.start)
    return rows


# Return the sound rows that are active at the current video time.
def pick_active(rows: list[SoundRow], t: float) -> list[SoundRow]:
    active: list[SoundRow] = []
    for row in rows:
        if row.start <= t <= row.stop:
            active.append(row)
        if row.start > t:
            break
    return active


# Read a WAV file and compress it into a normalized low-rate envelope for waveform visualization.
def load_waveform_envelope(wav_path: Path | None, target_hz: int = 200) -> tuple[np.ndarray | None, float | None, str]:
    if wav_path is None or not wav_path.exists():
        return None, None, "waveform unavailable: wav not found"

    with wave.open(str(wav_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw = wf.readframes(num_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None, None, f"waveform unavailable: unsupported sample width {sample_width}"

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    audio = np.abs(audio)
    if audio.size == 0 or sample_rate <= 0:
        return None, None, "waveform unavailable: empty audio"

    samples_per_bin = max(1, int(sample_rate / target_hz))
    envelope_size = int(np.ceil(audio.size / samples_per_bin))
    pad = envelope_size * samples_per_bin - audio.size
    if pad > 0:
        audio = np.pad(audio, (0, pad), mode="constant")
    envelope = audio.reshape(envelope_size, samples_per_bin).max(axis=1)

    max_value = float(np.max(envelope))
    if max_value > 1e-6:
        envelope = envelope / max_value

    return envelope.astype(np.float32), float(target_hz), f"waveform loaded from {wav_path}"


# Pre-render the full audio timeline panel so each frame only needs to crop the visible window.
def build_timeline_panel_canvas(
    rows: list[SoundRow],
    waveform_envelope: np.ndarray | None,
    waveform_hz: float | None,
    duration: float,
    panel_width: int,
    panel_height: int,
    timeline_window_seconds: float,
) -> tuple[np.ndarray, float]:
    pixels_per_second = max(1.0, panel_width / max(1e-6, timeline_window_seconds))
    canvas_width = max(panel_width + 2, int(np.ceil(duration * pixels_per_second)) + 2)
    canvas = np.full((panel_height, canvas_width, 3), 22, dtype=np.uint8)

    fill_y0 = int(0.14 * panel_height)
    fill_y1 = panel_height - int(0.14 * panel_height)
    center_y = int((fill_y0 + fill_y1) / 2)
    wave_amp = max(4, int((fill_y1 - fill_y0) * 0.45))

    seg_radius = max(5, int((fill_y1 - fill_y0) * 0.35))
    for row in rows:
        sx0 = int(row.start * pixels_per_second)
        sx1 = int(row.stop * pixels_per_second)
        sx0 = max(0, min(canvas_width - 1, sx0))
        sx1 = max(0, min(canvas_width - 1, sx1))
        if sx1 <= sx0:
            sx1 = min(canvas_width - 1, sx0 + 1)

        color = class_id_to_color_bgr(row.class_id)
        draw_rounded_rect(canvas, sx0, fill_y0, sx1, fill_y1, seg_radius, color, alpha=0.34)
        cv2.rectangle(canvas, (sx0, fill_y0), (sx1, fill_y1), color, 1, cv2.LINE_AA)

    if waveform_envelope is not None and waveform_hz is not None and waveform_envelope.size > 0:
        x_positions = np.arange(canvas_width, dtype=np.int32)
        times = x_positions / pixels_per_second
        envelope_indices = np.clip((times * waveform_hz).astype(np.int32), 0, waveform_envelope.size - 1)
        amplitudes = waveform_envelope[envelope_indices]

        points_top: list[tuple[int, int]] = []
        points_bottom: list[tuple[int, int]] = []
        for x, amp in zip(x_positions.tolist(), amplitudes.tolist()):
            amp_px = int(amp * wave_amp)
            points_top.append((x, center_y - amp_px))
            points_bottom.append((x, center_y + amp_px))

        polygon = np.array(points_top + points_bottom[::-1], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], (230, 230, 230), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.polylines(canvas, [np.array(points_top, dtype=np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(points_bottom, dtype=np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, center_y), (canvas_width - 1, center_y), (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.line(canvas, (0, center_y), (canvas_width - 1, center_y), (120, 120, 120), 1, cv2.LINE_AA)

    return canvas, pixels_per_second


# Draw the scrolling audio waveform and EPIC label panel at the bottom of a rendered frame.
def render_audio_panel(
    frame: np.ndarray,
    t: float,
    duration: float,
    rows: list[SoundRow],
    timeline_canvas: np.ndarray | None,
    timeline_pixels_per_second: float | None,
    waveform_envelope: np.ndarray | None,
    waveform_hz: float | None,
    font: int,
    font_scale: float,
    thickness: int,
    text_style: dict[str, int],
    timeline_window_seconds: float = 15.0,
) -> None:
    if duration <= 0:
        return

    h, w = frame.shape[:2]
    ui_scale = max(0.35, h / 720.0)
    bottom_bar_height = max(scaled_ui_value(70, ui_scale, 28), int(0.14 * h))
    bar_y1 = h - 1
    bar_y0 = h - bottom_bar_height
    panel_margin_x = max(12, int(0.02 * w))
    panel_x0 = panel_margin_x
    panel_x1 = w - panel_margin_x
    panel_radius = max(8, int((bar_y1 - bar_y0) * 0.24))

    draw_rounded_rect(frame, panel_x0, bar_y0, panel_x1, bar_y1, panel_radius, (22, 22, 22), alpha=0.75)

    n_grid = 8
    for i in range(n_grid + 1):
        gx = int(panel_x0 + (i / n_grid) * (panel_x1 - panel_x0))
        cv2.line(frame, (gx, bar_y0 + 2), (gx, bar_y1 - 2), (58, 58, 58), 1, cv2.LINE_AA)

    if timeline_canvas is not None and timeline_pixels_per_second is not None:
        window_end = min(duration, max(0.0, t))
        window_start = max(0.0, window_end - timeline_window_seconds)
        crop_x0 = max(0, int(window_start * timeline_pixels_per_second))
        crop_x1 = min(
            timeline_canvas.shape[1],
            max(crop_x0 + 2, int(np.ceil(window_end * timeline_pixels_per_second)) + 1),
        )
        crop = timeline_canvas[:, crop_x0:crop_x1]
        bar_height = max(1, bar_y1 - bar_y0)
        panel_w = max(1, panel_x1 - panel_x0)
        if crop.size > 0:
            panel_img = cv2.resize(crop, (panel_w, bar_height), interpolation=cv2.INTER_LINEAR)
            roi = frame[bar_y0:bar_y1, panel_x0:panel_x1]
            if roi.shape[:2] == panel_img.shape[:2]:
                cv2.addWeighted(panel_img, 0.92, roi, 0.08, 0, roi)

    cv2.line(frame, (panel_x1 - 1, bar_y0), (panel_x1 - 1, bar_y1), (255, 255, 255), 2, cv2.LINE_AA)

    active_rows = pick_active(rows, t)
    active_class_ids = sorted({row.class_id for row in active_rows})
    status_text = "Audio waveform"
    if waveform_envelope is None or waveform_hz is None or waveform_envelope.size == 0:
        status_text = "Waveform unavailable"
    if rows:
        status_text = f"{status_text} + EPIC labels"
    draw_centered_text_line(
        frame=frame,
        y_baseline=bar_y0 - 46,
        parts=[(status_text, (235, 235, 235))],
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=text_style["pad_x"],
        pad_y=text_style["pad_y"],
        gap_px=text_style["gap_px"],
        bg_alpha=0.58,
        radius=text_style["radius"],
    )

    active_parts: list[tuple[str, tuple[int, int, int]]] = [("Active class_ids:", (235, 235, 235))]
    if active_class_ids:
        active_parts.append((" ", (235, 235, 235)))
        for idx, class_id in enumerate(active_class_ids):
            suffix = "," if idx < len(active_class_ids) - 1 else ""
            active_parts.append((f"{class_id}{suffix}", class_id_to_color_bgr(class_id)))
    else:
        active_parts.append((" (none)", (200, 200, 200)))

    draw_centered_text_line(
        frame=frame,
        y_baseline=bar_y0 - 10,
        parts=active_parts,
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=text_style["pad_x"],
        pad_y=text_style["pad_y"],
        gap_px=text_style["gap_px"],
        bg_alpha=0.58,
        radius=text_style["radius"],
    )

    latest_desc = ""
    latest_class = ""
    latest_id = None
    if active_rows:
        latest = max(active_rows, key=lambda row: row.start)
        latest_desc = latest.description
        latest_class = latest.class_name
        latest_id = latest.class_id

    if latest_desc and latest_class:
        latest_text = f"Latest Description: {latest_desc} Latest Class: {latest_class} (class_id={latest_id})"
        latest_color = class_id_to_color_bgr(latest_id if latest_id is not None else 0)
    else:
        latest_text = "Latest: (none)"
        latest_color = (220, 220, 220)

    draw_centered_text_line(
        frame=frame,
        y_baseline=max(24, bar_y0 - 82),
        parts=[(latest_text, latest_color)],
        font=font,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=text_style["pad_x"],
        pad_y=text_style["pad_y"],
        gap_px=text_style["gap_px"],
        bg_alpha=0.58,
        radius=text_style["radius"],
    )


# Extract mono 16 kHz WAV audio from a video when the waveform overlay needs an audio source.
def extract_audio_to_wav(audio_source: Path, output_dir: Path) -> Path | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_dir, prefix="waveform_", suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(audio_source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass
        return None
    return wav_path


# Remove empty prompts and ensure the backend always receives at least one valid class prompt.
def clean_prompts(prompts: list[str]) -> list[str]:
    values = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not values:
        raise ValueError("At least one non-empty prompt is required.")
    return values


# Load the requested prompted-vision backend and configure it with the text prompts.
def load_backend(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    prompts = clean_prompts(args.prompts)

    if args.backend == "yoloe":
        from ultralytics import YOLOE

        model = YOLOE(args.model)
        model.set_classes(prompts, model.get_text_pe(prompts))
        return {"backend": "yoloe", "model": model}, prompts

    from ultralytics import FastSAM, YOLOWorld

    world = YOLOWorld(args.world_model)
    world.set_classes(prompts)
    fastsam = FastSAM(args.fastsam_model)
    return {"backend": "fastsam-world", "world": world, "fastsam": fastsam}, prompts


# Resolve a class name from an Ultralytics names mapping or fall back to the raw class id.
def name_from_lookup(names: Any, cls_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_idx, cls_idx))
    if isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
        return str(names[cls_idx])
    return str(cls_idx)


# Convert a binary mask into polygon contours for serialization and later rendering.
def mask_to_polygons(mask: np.ndarray) -> list[list[list[int]]]:
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[int]]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append(contour.reshape(-1, 2).astype(int).tolist())
    return polygons


# Extract image-space polygons from an Ultralytics result, falling back to raw mask contours if needed.
def polygons_from_result_masks(result, idx: int) -> list[list[list[int]]]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    xy_polygons = getattr(masks, "xy", None)
    if xy_polygons is not None and idx < len(xy_polygons):
        polygon_entry = xy_polygons[idx]
        polygon_arrays: list[np.ndarray] = []
        polygon_np = np.asarray(polygon_entry, dtype=np.float32)
        if polygon_np.ndim == 2 and polygon_np.shape[0] >= 3 and polygon_np.shape[1] == 2:
            polygon_arrays.append(polygon_np)
        else:
            for polygon in polygon_entry:
                polygon_np = np.asarray(polygon, dtype=np.float32)
                if polygon_np.ndim != 2 or polygon_np.shape[0] < 3 or polygon_np.shape[1] != 2:
                    continue
                polygon_arrays.append(polygon_np)

        polygons: list[list[list[int]]] = []
        for polygon_np in polygon_arrays:
            polygons.append(np.rint(polygon_np).astype(int).tolist())
        if polygons:
            return polygons

    mask_data = getattr(masks, "data", None)
    if mask_data is None:
        return []
    mask_array = mask_data.detach().cpu().numpy()
    if idx >= len(mask_array):
        return []
    return mask_to_polygons(mask_array[idx])


# Render a normalized list of cached detections as translucent masks, boxes, and labels.
def render_cached_detections(frame: np.ndarray, detections: list[dict[str, Any]], alpha: float) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()

    for det in detections:
        label = str(det.get("label", "object"))
        color = class_color_bgr(label)
        for polygon in det.get("polygons", []):
            if len(polygon) < 3:
                continue
            poly_np = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [poly_np], color, lineType=cv2.LINE_AA)

    if detections:
        annotated = cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0.0)

    for det in detections:
        label = str(det.get("label", "object"))
        color = class_color_bgr(label)
        box = det.get("box")
        score = float(det.get("score", 0.0))
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_label(annotated, f"{label} {score:.2f}", x1, y1, color)

    return annotated


# Normalize a YOLOE result into the project's standard detection dictionary format.
def build_yoloe_detections(result) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = getattr(result, "names", {})
    for idx, box in enumerate(boxes_xyxy):
        label = name_from_lookup(names, classes[idx])
        polygons = polygons_from_result_masks(result, idx)
        detections.append(
            {
                "label": label,
                "score": float(scores[idx]),
                "box": [float(v) for v in box.tolist()],
                "polygons": polygons,
            }
        )
    return detections


# Combine YOLO-World boxes with FastSAM masks into the same normalized detection format.
def build_fastsam_world_detections(world_result, fastsam_result) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if world_result.boxes is None or len(world_result.boxes) == 0:
        return detections

    boxes_xyxy = world_result.boxes.xyxy.detach().cpu().numpy()
    scores = world_result.boxes.conf.detach().cpu().numpy()
    classes = world_result.boxes.cls.detach().cpu().numpy().astype(int)
    names = getattr(world_result, "names", {})
    for idx, box in enumerate(boxes_xyxy):
        label = name_from_lookup(names, classes[idx])
        polygons = polygons_from_result_masks(fastsam_result, idx) if fastsam_result is not None else []
        detections.append(
            {
                "label": label,
                "score": float(scores[idx]),
                "box": [float(v) for v in box.tolist()],
                "polygons": polygons,
            }
        )
    return detections


# Run the selected backend on one frame and return normalized detections.
def predict_detections(
    backend: dict[str, Any],
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
) -> list[dict[str, Any]]:
    if backend["backend"] == "yoloe":
        result = backend["model"].predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        return build_yoloe_detections(result)

    world_result = backend["world"].predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]

    if world_result.boxes is None or len(world_result.boxes) == 0:
        return []

    bboxes = world_result.boxes.xyxy.detach().cpu().numpy().tolist()
    fastsam_result = backend["fastsam"].predict(
        source=frame,
        bboxes=bboxes,
        imgsz=imgsz,
        device=device,
        retina_masks=True,
        verbose=False,
    )[0]
    return build_fastsam_world_detections(world_result, fastsam_result)


# Save per-frame detections so rendering can be repeated later without rerunning inference.
def save_prediction_cache(
    cache_path: Path,
    frames: list[list[dict[str, Any]]],
    backend_name: str,
    prompts: list[str],
) -> None:
    ensure_parent_dir(cache_path)
    payload = {
        "version": 1,
        "backend": backend_name,
        "prompts": prompts,
        "frames": frames,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


# Load a previously saved detection cache from JSON.
def load_prediction_cache(cache_path: Path) -> dict[str, Any]:
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Draw a filled label box for one detection near its bounding box.
def draw_label(frame: np.ndarray, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_text = max(0, y1 - text_height - baseline - 6)
    cv2.rectangle(frame, (x1, y_text), (x1 + text_width + 8, y_text + text_height + baseline + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y_text + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# Convenience helper that predicts detections and immediately renders them onto a frame.
def predict_annotated_frame(
    backend: dict[str, Any],
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    device: str | None,
    mask_alpha: float,
) -> np.ndarray:
    detections = predict_detections(backend, frame, conf, imgsz, device)
    return render_cached_detections(frame, detections, alpha=mask_alpha)


# Optionally dump rendered frames as individual JPEGs alongside the output video.
def save_frame_if_requested(save_frames: bool, output_path: Path, frame_index: int, frame: np.ndarray) -> None:
    if not save_frames:
        return

    frames_dir = output_path.with_suffix("")
    frames_dir = frames_dir.parent / f"{frames_dir.name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(frame_path), frame)


# Process a video file frame by frame, run prompted detection, render the overlays, and save a video.
def run_on_video(
    backend: dict[str, Any],
    input_path: Path,
    output_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    mask_alpha: float,
    max_seconds: float | None,
    save_frames: bool,
    copy_audio: bool,
    audio_panel: dict[str, Any] | None,
    cached_frames: list[list[dict[str, Any]]] | None,
    save_predictions_path: Path | None,
    backend_name: str,
    prompts: list[str],
    resize_factor: float,
    frame_stride: int,
) -> tuple[Path, bool, str | None]:
    fps, frame_count, width, height, duration = probe_video_with_ffmpeg(input_path)
    out_width, out_height = compute_output_size(width, height, resize_factor)

    silent_output_path = output_path
    final_output_path = output_path
    if copy_audio:
        silent_output_path = output_path.with_name(f"{output_path.stem}.silent{output_path.suffix}")

    reader = FfmpegVideoReader(input_path, width=width, height=height)
    writer = create_writer(silent_output_path, fps=fps, width=out_width, height=out_height)

    max_frames = None
    if max_seconds is not None:
        max_frames = max(1, int(round(max_seconds * fps)))

    try:
        frame_index = 0
        saved_frames: list[list[dict[str, Any]]] = []
        last_detections: list[dict[str, Any]] = []
        while True:
            ok, frame = reader.read()
            if not ok:
                break

            if max_frames is not None and frame_index >= max_frames:
                break

            if resize_factor != 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            if cached_frames is not None:
                detections = cached_frames[frame_index] if frame_index < len(cached_frames) else []
            else:
                should_infer = frame_index == 0 or max(1, frame_stride) <= 1 or (frame_index % max(1, frame_stride) == 0)
                if should_infer:
                    detections = predict_detections(
                        backend=backend,
                        frame=frame,
                        conf=conf,
                        imgsz=imgsz,
                        device=device,
                    )
                    last_detections = detections
                else:
                    detections = last_detections
                if save_predictions_path is not None:
                    saved_frames.append(detections)

            annotated = render_cached_detections(frame, detections, alpha=mask_alpha)
            if audio_panel is not None:
                render_audio_panel(
                    frame=annotated,
                    t=frame_index / fps,
                    duration=audio_panel["duration"],
                    rows=audio_panel["rows"],
                    timeline_canvas=audio_panel["timeline_canvas"],
                    timeline_pixels_per_second=audio_panel["timeline_pixels_per_second"],
                    waveform_envelope=audio_panel["waveform_envelope"],
                    waveform_hz=audio_panel["waveform_hz"],
                    font=audio_panel["font"],
                    font_scale=audio_panel["font_scale"],
                    thickness=audio_panel["thickness"],
                    text_style=audio_panel["text_style"],
                )
            writer.write(annotated)
            save_frame_if_requested(save_frames, output_path, frame_index, annotated)

            frame_index += 1
            if frame_index % 25 == 0:
                total = min(frame_count, max_frames) if max_frames is not None and frame_count > 0 else frame_count
                if total > 0:
                    print(f"Processed {frame_index}/{total} frames")
                else:
                    print(f"Processed {frame_index} frames")
    finally:
        reader.release()
        writer.release()

    if save_predictions_path is not None and cached_frames is None:
        save_prediction_cache(save_predictions_path, saved_frames, backend_name=backend_name, prompts=prompts)
        print(f"Saved predictions cache to: {save_predictions_path}")

    if not copy_audio:
        return final_output_path, False, None

    processed_duration = duration
    if max_frames is not None:
        processed_duration = max_frames / fps

    muxed, message = mux_audio_into_video(
        silent_video_path=silent_output_path,
        output_path=final_output_path,
        audio_source=input_path,
        duration_seconds=processed_duration,
    )
    if muxed:
        try:
            silent_output_path.unlink()
        except OSError:
            pass
        return final_output_path, True, message

    return silent_output_path, False, message


# Process a directory of image frames with the same rendering pipeline used for videos.
def run_on_frames_dir(
    backend: dict[str, Any],
    input_path: Path,
    output_path: Path,
    conf: float,
    imgsz: int,
    device: str | None,
    fps: float,
    mask_alpha: float,
    max_seconds: float | None,
    save_frames: bool,
    audio_panel: dict[str, Any] | None,
    cached_frames: list[list[dict[str, Any]]] | None,
    save_predictions_path: Path | None,
    backend_name: str,
    prompts: list[str],
    resize_factor: float,
    frame_stride: int,
) -> None:
    frame_paths = collect_frame_paths(input_path)
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    out_width, out_height = compute_output_size(width, height, resize_factor)
    writer = create_writer(output_path, fps=fps, width=out_width, height=out_height)

    max_frames = len(frame_paths)
    if max_seconds is not None:
        max_frames = min(max_frames, max(1, int(round(max_seconds * fps))))

    try:
        saved_frames: list[list[dict[str, Any]]] = []
        last_detections: list[dict[str, Any]] = []
        for frame_index, frame_path in enumerate(frame_paths[:max_frames]):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")

            if resize_factor != 1.0:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            if cached_frames is not None:
                detections = cached_frames[frame_index] if frame_index < len(cached_frames) else []
            else:
                should_infer = frame_index == 0 or max(1, frame_stride) <= 1 or (frame_index % max(1, frame_stride) == 0)
                if should_infer:
                    detections = predict_detections(
                        backend=backend,
                        frame=frame,
                        conf=conf,
                        imgsz=imgsz,
                        device=device,
                    )
                    last_detections = detections
                else:
                    detections = last_detections
                if save_predictions_path is not None:
                    saved_frames.append(detections)

            annotated = render_cached_detections(frame, detections, alpha=mask_alpha)
            if audio_panel is not None:
                render_audio_panel(
                    frame=annotated,
                    t=frame_index / fps,
                    duration=audio_panel["duration"],
                    rows=audio_panel["rows"],
                    timeline_canvas=audio_panel["timeline_canvas"],
                    timeline_pixels_per_second=audio_panel["timeline_pixels_per_second"],
                    waveform_envelope=audio_panel["waveform_envelope"],
                    waveform_hz=audio_panel["waveform_hz"],
                    font=audio_panel["font"],
                    font_scale=audio_panel["font_scale"],
                    thickness=audio_panel["thickness"],
                    text_style=audio_panel["text_style"],
                )
            writer.write(annotated)
            save_frame_if_requested(save_frames, output_path, frame_index, annotated)

            if (frame_index + 1) % 25 == 0:
                print(f"Processed {frame_index + 1}/{max_frames} frames")
    finally:
        writer.release()

    if save_predictions_path is not None and cached_frames is None:
        save_prediction_cache(save_predictions_path, saved_frames, backend_name=backend_name, prompts=prompts)
        print(f"Saved predictions cache to: {save_predictions_path}")


# Wire together input loading, backend setup, optional audio overlays, rendering, and final output writing.
def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    cached_payload = None
    cached_frames = None
    if args.load_predictions is not None:
        cached_payload = load_prediction_cache(args.load_predictions.resolve())
        cached_frames = cached_payload.get("frames", [])
        backend = {"backend": "cached"}
        prompts = cached_payload.get("prompts", [])
    else:
        backend, prompts = load_backend(args)
    audio_panel = None
    temp_audio_wav = None

    print(f"Backend: {'cached' if cached_payload is not None else args.backend}")
    if cached_payload is not None:
        print(f"Loaded predictions cache: {args.load_predictions.resolve()}")
    elif args.backend == "yoloe":
        print(f"Loaded model: {args.model}")
    else:
        print(f"Loaded YOLO-World model: {args.world_model}")
        print(f"Loaded FastSAM model: {args.fastsam_model}")
    print(f"Using prompts: {prompts}")
    print(f"Frame stride: {max(1, args.frame_stride)}")
    if args.max_seconds is not None:
        print(
            f"Preview mode enabled via --max-seconds={args.max_seconds}. "
            "Only the first part of the input will be processed."
        )

    if args.visualize_audio:
        video_id = args.video_id.strip()
        if not video_id and input_path.is_file():
            video_id = input_path.stem

        probe_duration = None
        if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
            probe_fps, probe_frame_count, _probe_width, _probe_height, probe_duration = probe_video_with_ffmpeg(input_path)
            if args.max_seconds is not None:
                max_frames = max(1, int(round(args.max_seconds * probe_fps)))
                probe_duration = min(probe_duration, max_frames / probe_fps)
        elif input_path.is_dir():
            frame_paths = collect_frame_paths(input_path)
            frame_count = len(frame_paths)
            probe_duration = frame_count / args.fps
            if args.max_seconds is not None:
                probe_duration = min(probe_duration, max(1, int(round(args.max_seconds * args.fps))) / args.fps)

        waveform_path = args.audio_source.resolve() if args.audio_source else None
        if waveform_path is None and input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
            temp_audio_wav = extract_audio_to_wav(input_path, output_path.parent)
            waveform_path = temp_audio_wav

        waveform_envelope, waveform_hz, waveform_status = load_waveform_envelope(waveform_path)
        train_rows = load_sound_rows(args.train_csv.resolve(), video_id) if video_id else []
        val_rows = load_sound_rows(args.val_csv.resolve(), video_id) if video_id else []
        if args.csv_split == "train":
            sound_rows = train_rows
            source_csv = "train"
        elif args.csv_split == "validation":
            sound_rows = val_rows
            source_csv = "validation"
        else:
            sound_rows = train_rows if train_rows else val_rows
            source_csv = "train" if train_rows else "validation"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        text_style = scaled_text_style(1.0)
        timeline_canvas = None
        timeline_pixels_per_second = None
        timeline_panel_width = max(640, int(round(1280 * max(0.35, args.resize_factor))))
        timeline_panel_height = max(80, int(round(160 * max(0.35, args.resize_factor))))
        if probe_duration is not None:
            timeline_canvas, timeline_pixels_per_second = build_timeline_panel_canvas(
                rows=sound_rows,
                waveform_envelope=waveform_envelope,
                waveform_hz=waveform_hz,
                duration=probe_duration,
                panel_width=timeline_panel_width,
                panel_height=timeline_panel_height,
                timeline_window_seconds=15.0,
            )
        audio_panel = {
            "duration": probe_duration or 0.0,
            "rows": sound_rows,
            "timeline_canvas": timeline_canvas,
            "timeline_pixels_per_second": timeline_pixels_per_second,
            "waveform_envelope": waveform_envelope,
            "waveform_hz": waveform_hz,
            "font": font,
            "font_scale": max(0.35, 0.75 * max(0.35, args.resize_factor)),
            "thickness": scaled_ui_value(2, max(0.35, args.resize_factor), 1),
            "text_style": scaled_text_style(max(0.35, args.resize_factor)),
        }
        print(f"Waveform: {waveform_status}")
        if video_id:
            print(f"EPIC video_id: {video_id}")
            print(f"Loaded {len(sound_rows)} sound rows from {source_csv} CSV")
    print(f"Resize factor: {args.resize_factor}")

    if input_path.is_dir():
        run_on_frames_dir(
            backend=backend,
            input_path=input_path,
            output_path=output_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            fps=args.fps,
            mask_alpha=args.mask_alpha,
            max_seconds=args.max_seconds,
            save_frames=args.save_frames,
            audio_panel=audio_panel,
            cached_frames=cached_frames,
            save_predictions_path=args.save_predictions.resolve() if args.save_predictions else None,
            backend_name=("cached" if cached_payload is not None else args.backend),
            prompts=prompts,
            resize_factor=args.resize_factor,
            frame_stride=max(1, args.frame_stride),
        )
    elif input_path.suffix.lower() in VIDEO_EXTS:
        written_output_path, audio_muxed, audio_message = run_on_video(
            backend=backend,
            input_path=input_path,
            output_path=output_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            mask_alpha=args.mask_alpha,
            max_seconds=args.max_seconds,
            save_frames=args.save_frames,
            copy_audio=args.copy_audio,
            audio_panel=audio_panel,
            cached_frames=cached_frames,
            save_predictions_path=args.save_predictions.resolve() if args.save_predictions else None,
            backend_name=("cached" if cached_payload is not None else args.backend),
            prompts=prompts,
            resize_factor=args.resize_factor,
            frame_stride=max(1, args.frame_stride),
        )
    else:
        raise ValueError(
            f"Unsupported input path: {input_path}. Provide a video file or a directory of image frames."
        )

    if temp_audio_wav is not None:
        try:
            temp_audio_wav.unlink()
        except OSError:
            pass

    if input_path.suffix.lower() in VIDEO_EXTS:
        if args.copy_audio:
            if audio_muxed:
                print(f"Audio: {audio_message}")
            else:
                print(f"Audio mux skipped or failed: {audio_message}")
        print(f"Saved rendered video to: {written_output_path}")
    else:
        print(f"Saved rendered video to: {output_path}")


if __name__ == "__main__":
    main()
