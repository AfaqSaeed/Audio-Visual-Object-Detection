from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize temporal carrot/cucumber evidence scores from a split_continuous_water_events JSON "
            "as a static PNG timeline."
        )
    )
    parser.add_argument("--split-json", type=Path, required=True, help="Path to *_split_events.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to *_scores.png next to the split JSON.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    split_path = args.split_json.resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_path}")

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    frame_scores: list[dict[str, Any]] = payload.get("frame_scores", [])
    segments: list[dict[str, Any]] = payload.get("segments", [])
    if not frame_scores:
        raise RuntimeError(f"No frame_scores found in {split_path}")

    output_path = args.output.resolve() if args.output else split_path.with_name(f"{split_path.stem}_scores.png")

    width = 1800
    height = 980
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 80
    chart_h = 320
    gap_h = 70

    chart1_y0 = margin_top
    chart1_y1 = chart1_y0 + chart_h
    chart2_y0 = chart1_y1 + gap_h
    chart2_y1 = chart2_y0 + chart_h
    band_y0 = chart2_y1 + 40
    band_y1 = height - margin_bottom

    image = np.full((height, width, 3), 246, dtype=np.uint8)
    plot_x0 = margin_left
    plot_x1 = width - margin_right
    plot_w = plot_x1 - plot_x0

    carrot_raw = np.array([float(item.get("carrot_raw", 0.0)) for item in frame_scores], dtype=np.float32)
    cucumber_raw = np.array([float(item.get("cucumber_raw", 0.0)) for item in frame_scores], dtype=np.float32)
    carrot_smooth = np.array([float(item.get("carrot_smooth", 0.0)) for item in frame_scores], dtype=np.float32)
    cucumber_smooth = np.array([float(item.get("cucumber_smooth", 0.0)) for item in frame_scores], dtype=np.float32)
    dominant_labels = [item.get("dominant_label") for item in frame_scores]

    max_score = float(max(np.max(carrot_raw), np.max(cucumber_raw), np.max(carrot_smooth), np.max(cucumber_smooth), 0.1))
    fps = float(payload.get("fps_estimate", 30.0))
    clip_start_sec = float(payload.get("clip_start_sec", 0.0))

    def x_for_index(index: int) -> int:
        if len(frame_scores) <= 1:
            return plot_x0
        return int(round(plot_x0 + (index / (len(frame_scores) - 1)) * plot_w))

    def y_for_score(score: float, y0: int, y1: int) -> int:
        usable_h = y1 - y0
        return int(round(y1 - (score / max_score) * usable_h))

    def draw_chart(y0: int, y1: int, raw_values: np.ndarray, smooth_values: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(image, (plot_x0, y0), (plot_x1, y1), (210, 210, 210), 1, cv2.LINE_AA)
        for i in range(6):
            gy = int(round(y0 + (i / 5.0) * (y1 - y0)))
            cv2.line(image, (plot_x0, gy), (plot_x1, gy), (228, 228, 228), 1, cv2.LINE_AA)
            score_value = max_score * (1.0 - (i / 5.0))
            draw_text(image, f"{score_value:.2f}", 18, gy + 4, (110, 110, 110), scale=0.45, thickness=1)

        raw_points = np.array(
            [[x_for_index(idx), y_for_score(float(value), y0, y1)] for idx, value in enumerate(raw_values)],
            dtype=np.int32,
        )
        smooth_points = np.array(
            [[x_for_index(idx), y_for_score(float(value), y0, y1)] for idx, value in enumerate(smooth_values)],
            dtype=np.int32,
        )
        if len(raw_points) > 1:
            cv2.polylines(image, [raw_points], False, tuple(int(c * 0.55) for c in color), 1, cv2.LINE_AA)
        if len(smooth_points) > 1:
            cv2.polylines(image, [smooth_points], False, color, 3, cv2.LINE_AA)

        draw_text(image, f"{label} raw", plot_x0 + 10, y0 - 18, tuple(int(c * 0.55) for c in color), scale=0.55, thickness=1)
        draw_text(image, f"{label} smooth", plot_x0 + 130, y0 - 18, color, scale=0.55, thickness=2)

    draw_text(image, "Water Segment Split Debug", plot_x0, 36, (30, 30, 30), scale=1.0, thickness=2)
    draw_text(image, f"clip: {clip_start_sec:.2f}s -> {float(payload.get('clip_end_sec', 0.0)):.2f}s", plot_x0, 62, (70, 70, 70), scale=0.55, thickness=1)
    draw_chart(chart1_y0, chart1_y1, carrot_raw, carrot_smooth, "carrot", (40, 120, 255))
    draw_chart(chart2_y0, chart2_y1, cucumber_raw, cucumber_smooth, "cucumber", (40, 180, 90))

    cv2.rectangle(image, (plot_x0, band_y0), (plot_x1, band_y1), (210, 210, 210), 1, cv2.LINE_AA)
    draw_text(image, "dominant split segments", plot_x0 + 10, band_y0 - 16, (60, 60, 60), scale=0.58, thickness=1)

    label_to_color = {
        None: (230, 230, 230),
        "washing carrot": (40, 120, 255),
        "washing cucumber": (40, 180, 90),
    }
    for idx, label in enumerate(dominant_labels):
        x = x_for_index(idx)
        next_x = x_for_index(min(idx + 1, len(dominant_labels) - 1))
        color = label_to_color.get(label, (170, 170, 170))
        cv2.rectangle(image, (x, band_y0 + 8), (max(x + 1, next_x), band_y1 - 8), color, -1)

    for seg in segments:
        start_frame = int(seg["start_frame"])
        end_frame = int(seg["end_frame"])
        start_x = x_for_index(start_frame)
        end_x = x_for_index(end_frame)
        label = str(seg["label"])
        color = label_to_color.get(label, (170, 170, 170))
        cv2.rectangle(image, (start_x, band_y0 + 8), (max(start_x + 2, end_x), band_y1 - 8), color, -1)
        draw_text(
            image,
            f"{label}: {seg['start_sec_in_video']:.2f}s -> {seg['end_sec_in_video']:.2f}s",
            start_x + 6,
            band_y0 + 34,
            (255, 255, 255),
            scale=0.5,
            thickness=1,
        )

    tick_count = 10
    for i in range(tick_count + 1):
        frame_index = int(round((i / tick_count) * (len(frame_scores) - 1)))
        x = x_for_index(frame_index)
        cv2.line(image, (x, chart1_y1), (x, band_y1 + 6), (180, 180, 180), 1, cv2.LINE_AA)
        sec_in_video = clip_start_sec + (frame_index / max(fps, 1e-6))
        draw_text(image, f"{sec_in_video:.1f}s", x - 18, band_y1 + 28, (90, 90, 90), scale=0.45, thickness=1)

    params = payload.get("params", {})
    recipe_lines = [
        f"smooth={params.get('smooth_window_frames', '?')}f",
        f"start={params.get('start_support_frames', '?')}f",
        f"end_gap={params.get('end_gap_frames', '?')}f",
        f"margin={params.get('dominance_margin', '?')}",
    ]
    draw_text(image, " | ".join(recipe_lines), plot_x0, height - 18, (70, 70, 70), scale=0.52, thickness=1)

    ensure_parent(output_path)
    cv2.imwrite(str(output_path), image)
    print(f"Saved split score visualization to: {output_path}")


if __name__ == "__main__":
    main()
