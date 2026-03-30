from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectionCandidate:
    label: str
    score: float
    center_x: float
    center_y: float


@dataclass(frozen=True)
class EventRecipe:
    name: str
    required_groups: list[list[str]]
    required_group_min_scores: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split one continuous water event clip into visual sub-events such as washing carrot "
            "and washing cucumber using per-frame evidence plus hysteresis."
        )
    )
    parser.add_argument("--report-json", type=Path, required=True, help="Pipeline report JSON for the parent water clip.")
    parser.add_argument("--predictions-json", type=Path, default=None, help="Optional predictions JSON override.")
    parser.add_argument("--carrot-recipe", type=Path, required=True, help="Recipe JSON for washing carrot.")
    parser.add_argument("--cucumber-recipe", type=Path, required=True, help="Recipe JSON for washing cucumber.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON for split sub-events.")
    parser.add_argument(
        "--smooth-window-frames",
        type=int,
        default=15,
        help="Centered moving-average window for stabilizing frame evidence.",
    )
    parser.add_argument(
        "--start-support-frames",
        type=int,
        default=5,
        help="How many consecutive frames an event must dominate before a segment starts.",
    )
    parser.add_argument(
        "--end-gap-frames",
        type=int,
        default=10,
        help="How many consecutive non-dominant frames are needed before a segment ends.",
    )
    parser.add_argument(
        "--dominance-margin",
        type=float,
        default=0.03,
        help="Minimum smoothed score gap required for one event to dominate the other.",
    )
    parser.add_argument(
        "--distance-threshold-ratio",
        type=float,
        default=0.08,
        help="Fallback spatial threshold ratio if not found in the parent report.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_required_groups(raw_groups: list[Any]) -> list[list[str]]:
    groups: list[list[str]] = []
    for group in raw_groups:
        if isinstance(group, str):
            groups.append([normalize_text(part) for part in group.split("|") if normalize_text(part)])
        else:
            groups.append([normalize_text(item) for item in group if normalize_text(item)])
    return groups


def parse_recipe(path: Path) -> EventRecipe:
    payload = load_json(path)
    required_groups = parse_required_groups(payload["must_have"])
    raw_scores = payload.get("must_have_min_scores", [])
    if raw_scores:
        required_group_min_scores = [float(value) for value in raw_scores]
    else:
        required_group_min_scores = [0.0] * len(required_groups)
    return EventRecipe(
        name=str(payload["positive_event_name"]),
        required_groups=required_groups,
        required_group_min_scores=required_group_min_scores,
    )


def center_from_detection(det: dict[str, Any]) -> tuple[float, float] | None:
    polygons = det.get("polygons") or []
    points: list[tuple[float, float]] = []
    for polygon in polygons:
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                points.append((float(point[0]), float(point[1])))
    if points:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )

    box = det.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = [float(value) for value in box]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return None


def build_group_candidates(
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
        candidate = DetectionCandidate(label=label, score=score, center_x=center[0], center_y=center[1])
        for group_index, aliases in enumerate(required_groups):
            if score < required_group_min_scores[group_index]:
                continue
            if any(alias in label or label in alias for alias in aliases):
                grouped[group_index].append(candidate)
    return grouped


def best_combo_score(
    frame_dets: list[dict[str, Any]],
    recipe: EventRecipe,
    distance_threshold_px: float,
) -> dict[str, Any]:
    grouped = build_group_candidates(frame_dets, recipe.required_groups, recipe.required_group_min_scores)
    if any(not group for group in grouped):
        return {
            "complete": False,
            "passed": False,
            "score": 0.0,
            "max_pairwise_distance": None,
            "avg_detection_score": 0.0,
        }

    best: dict[str, Any] | None = None
    for combo in __import__("itertools").product(*grouped):
        max_pairwise_distance = 0.0
        for left, right in __import__("itertools").combinations(combo, 2):
            max_pairwise_distance = max(
                max_pairwise_distance,
                math.hypot(left.center_x - right.center_x, left.center_y - right.center_y),
            )
        avg_detection_score = sum(item.score for item in combo) / len(combo)
        passed = max_pairwise_distance <= distance_threshold_px
        closeness_score = max(0.0, 1.0 - (max_pairwise_distance / max(distance_threshold_px, 1e-6)))
        composite_score = avg_detection_score * (0.25 + 0.75 * closeness_score) if passed else 0.0
        current = {
            "complete": True,
            "passed": passed,
            "score": composite_score,
            "max_pairwise_distance": max_pairwise_distance,
            "avg_detection_score": avg_detection_score,
        }
        if best is None:
            best = current
            continue
        if current["score"] > best["score"]:
            best = current
            continue
        if math.isclose(current["score"], best["score"]) and (
            current["max_pairwise_distance"] or float("inf")
        ) < (best["max_pairwise_distance"] or float("inf")):
            best = current

    assert best is not None
    return best


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values[:]
    radius = max(0, window // 2)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        stop = min(len(values), idx + radius + 1)
        chunk = values[start:stop]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def frames_to_segments(
    labels: list[str | None],
    fps: float,
    clip_offset_sec: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active_label: str | None = None
    start_index = 0
    for idx, label in enumerate(labels):
        if label == active_label:
            continue
        if active_label is not None:
            segments.append(
                {
                    "label": active_label,
                    "start_frame": start_index,
                    "end_frame": idx - 1,
                    "start_sec_in_clip": start_index / fps,
                    "end_sec_in_clip": idx / fps,
                    "start_sec_in_video": clip_offset_sec + (start_index / fps),
                    "end_sec_in_video": clip_offset_sec + (idx / fps),
                }
            )
        active_label = label
        start_index = idx

    if active_label is not None:
        segments.append(
            {
                "label": active_label,
                "start_frame": start_index,
                "end_frame": len(labels) - 1,
                "start_sec_in_clip": start_index / fps,
                "end_sec_in_clip": len(labels) / fps,
                "start_sec_in_video": clip_offset_sec + (start_index / fps),
                "end_sec_in_video": clip_offset_sec + (len(labels) / fps),
            }
        )
    return segments


def main() -> None:
    args = parse_args()
    report_path = args.report_json.resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    report = load_json(report_path)
    predictions_path = args.predictions_json.resolve() if args.predictions_json else Path(report["paths"]["predictions_json"])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions JSON not found: {predictions_path}")

    carrot_recipe = parse_recipe(args.carrot_recipe.resolve())
    cucumber_recipe = parse_recipe(args.cucumber_recipe.resolve())
    predictions = load_json(predictions_path)
    frames: list[list[dict[str, Any]]] = predictions.get("frames", [])

    analysis = report["analysis"]
    video_size = analysis.get("video_size", {})
    width = int(video_size.get("width", 1920))
    height = int(video_size.get("height", 1080))
    distance_threshold_px = float(
        analysis.get("distance_threshold_px", max(width, height) * float(args.distance_threshold_ratio))
    )

    clip_start_sec = float(report["audio_hit"]["clip_start_sec"])
    clip_end_sec = float(report["audio_hit"]["clip_end_sec"])
    clip_duration = max(1e-6, clip_end_sec - clip_start_sec)
    fps = len(frames) / clip_duration if frames else 30.0

    carrot_raw: list[float] = []
    cucumber_raw: list[float] = []
    frame_debug: list[dict[str, Any]] = []
    for frame_index, frame_dets in enumerate(frames):
        carrot_score = best_combo_score(frame_dets, carrot_recipe, distance_threshold_px)
        cucumber_score = best_combo_score(frame_dets, cucumber_recipe, distance_threshold_px)
        carrot_raw.append(float(carrot_score["score"]))
        cucumber_raw.append(float(cucumber_score["score"]))
        frame_debug.append(
            {
                "frame_index": frame_index,
                "carrot": carrot_score,
                "cucumber": cucumber_score,
            }
        )

    carrot_smooth = moving_average(carrot_raw, args.smooth_window_frames)
    cucumber_smooth = moving_average(cucumber_raw, args.smooth_window_frames)

    dominant_labels: list[str | None] = []
    active_label: str | None = None
    start_counter = 0
    end_counter = 0
    pending_label: str | None = None
    for carrot_value, cucumber_value in zip(carrot_smooth, cucumber_smooth):
        winner: str | None = None
        if carrot_value > 0 and carrot_value >= cucumber_value + args.dominance_margin:
            winner = carrot_recipe.name
        elif cucumber_value > 0 and cucumber_value >= carrot_value + args.dominance_margin:
            winner = cucumber_recipe.name

        if active_label is None:
            if winner is None:
                start_counter = 0
                pending_label = None
                dominant_labels.append(None)
                continue
            if winner != pending_label:
                pending_label = winner
                start_counter = 1
            else:
                start_counter += 1
            if start_counter >= args.start_support_frames:
                active_label = winner
                end_counter = 0
            dominant_labels.append(active_label)
            continue

        if winner == active_label:
            end_counter = 0
            dominant_labels.append(active_label)
            continue

        if winner is None:
            end_counter += 1
            if end_counter >= args.end_gap_frames:
                active_label = None
                pending_label = None
                start_counter = 0
            dominant_labels.append(active_label)
            continue

        end_counter += 1
        if end_counter >= args.end_gap_frames:
            active_label = winner
            pending_label = winner
            start_counter = args.start_support_frames
            end_counter = 0
        dominant_labels.append(active_label)

    segments = frames_to_segments(dominant_labels, fps=fps, clip_offset_sec=clip_start_sec)
    output_path = args.output_json.resolve() if args.output_json else report_path.with_name(f"{report_path.stem}_split_events.json")
    payload = {
        "report_json": str(report_path),
        "predictions_json": str(predictions_path),
        "fps_estimate": fps,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "distance_threshold_px": distance_threshold_px,
        "recipes": [
            {
                "name": carrot_recipe.name,
                "required_groups": carrot_recipe.required_groups,
                "required_group_min_scores": carrot_recipe.required_group_min_scores,
            },
            {
                "name": cucumber_recipe.name,
                "required_groups": cucumber_recipe.required_groups,
                "required_group_min_scores": cucumber_recipe.required_group_min_scores,
            },
        ],
        "params": {
            "smooth_window_frames": args.smooth_window_frames,
            "start_support_frames": args.start_support_frames,
            "end_gap_frames": args.end_gap_frames,
            "dominance_margin": args.dominance_margin,
        },
        "segments": segments,
        "frame_scores": [
            {
                "frame_index": item["frame_index"],
                "carrot_raw": carrot_raw[item["frame_index"]],
                "cucumber_raw": cucumber_raw[item["frame_index"]],
                "carrot_smooth": carrot_smooth[item["frame_index"]],
                "cucumber_smooth": cucumber_smooth[item["frame_index"]],
                "dominant_label": dominant_labels[item["frame_index"]],
            }
            for item in frame_debug
        ],
    }
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved split water sub-events to: {output_path}")


if __name__ == "__main__":
    main()
