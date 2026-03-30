from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare predicted EPIC-Sounds-style detections against ground truth train annotations."
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        required=True,
        help="Path to the predicted detections CSV.",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        required=True,
        help="Path to the EPIC_Sounds_train.csv ground-truth file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the per-detection comparison CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write summary metrics as JSON.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.1,
        help="Minimum IoU used for greedy one-to-one matching summary metrics.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_int(value: str, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Missing required integer field: {field_name}")
    return int(float(value))


def interval_metrics(a_start: int, a_stop: int, b_start: int, b_stop: int) -> tuple[int, int, int, float]:
    intersection = max(0, min(a_stop, b_stop) - max(a_start, b_start))
    a_len = max(0, a_stop - a_start)
    b_len = max(0, b_stop - b_start)
    union = a_len + b_len - intersection
    iou = 0.0 if union <= 0 else intersection / union
    return intersection, a_len, b_len, iou


def compare_rows(
    detection_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    gt_by_video: dict[str, list[dict[str, str]]] = {}
    for row in gt_rows:
        gt_by_video.setdefault(row["video_id"], []).append(row)

    comparison_rows: list[dict[str, object]] = []
    greedy_candidates: list[dict[str, object]] = []

    for det_idx, det in enumerate(detection_rows):
        video_id = det["video_id"]
        det_start = to_int(det["start_sample"], "start_sample")
        det_stop = to_int(det["stop_sample"], "stop_sample")
        det_class_id = to_int(det["class_id"], "class_id")
        candidates = gt_by_video.get(video_id, [])

        best_any = None
        best_same_class = None

        for gt_idx, gt in enumerate(candidates):
            gt_start = to_int(gt["start_sample"], "start_sample")
            gt_stop = to_int(gt["stop_sample"], "stop_sample")
            gt_class_id = to_int(gt["class_id"], "class_id")
            intersection, det_len, gt_len, iou = interval_metrics(det_start, det_stop, gt_start, gt_stop)
            if intersection <= 0:
                continue

            candidate = {
                "det_idx": det_idx,
                "gt_global_key": gt["annotation_id"],
                "video_id": video_id,
                "intersection_samples": intersection,
                "det_len": det_len,
                "gt_len": gt_len,
                "iou": iou,
                "class_match": det_class_id == gt_class_id,
                "gt_row": gt,
            }

            if best_any is None or candidate["iou"] > best_any["iou"]:
                best_any = candidate
            if candidate["class_match"] and (best_same_class is None or candidate["iou"] > best_same_class["iou"]):
                best_same_class = candidate

            if candidate["class_match"]:
                greedy_candidates.append(candidate)

        chosen = best_same_class if best_same_class is not None else best_any

        row = dict(det)
        row["matched_gt_annotation_id"] = chosen["gt_row"]["annotation_id"] if chosen else ""
        row["matched_gt_start_sample"] = chosen["gt_row"]["start_sample"] if chosen else ""
        row["matched_gt_stop_sample"] = chosen["gt_row"]["stop_sample"] if chosen else ""
        row["matched_gt_start_timestamp"] = chosen["gt_row"]["start_timestamp"] if chosen else ""
        row["matched_gt_stop_timestamp"] = chosen["gt_row"]["stop_timestamp"] if chosen else ""
        row["matched_gt_description"] = chosen["gt_row"]["description"] if chosen else ""
        row["matched_gt_class"] = chosen["gt_row"]["class"] if chosen else ""
        row["matched_gt_class_id"] = chosen["gt_row"]["class_id"] if chosen else ""
        row["overlap_found"] = bool(chosen)
        row["class_match"] = bool(chosen and chosen["class_match"])
        row["iou"] = round(chosen["iou"], 6) if chosen else 0.0
        row["intersection_samples"] = int(chosen["intersection_samples"]) if chosen else 0
        comparison_rows.append(row)

    greedy_candidates.sort(key=lambda item: item["iou"], reverse=True)
    matched_det_indices: set[int] = set()
    matched_gt_keys: set[str] = set()

    for candidate in greedy_candidates:
        if candidate["iou"] < 0.0:
            continue
        if candidate["iou"] < 0.0:
            continue
        det_idx = int(candidate["det_idx"])
        gt_key = str(candidate["gt_global_key"])
        if det_idx in matched_det_indices or gt_key in matched_gt_keys:
            continue
        if candidate["iou"] < 0.0:
            continue
        matched_det_indices.add(det_idx)
        matched_gt_keys.add(gt_key)

    greedy_matches_above_threshold = []
    matched_det_indices.clear()
    matched_gt_keys.clear()
    for candidate in greedy_candidates:
        if candidate["iou"] < 0.0:
            continue
        det_idx = int(candidate["det_idx"])
        gt_key = str(candidate["gt_global_key"])
        if det_idx in matched_det_indices or gt_key in matched_gt_keys:
            continue
        greedy_matches_above_threshold.append(candidate)
        matched_det_indices.add(det_idx)
        matched_gt_keys.add(gt_key)

    summary = {
        "num_detections": len(detection_rows),
        "num_ground_truth": len(gt_rows),
        "detections_with_any_overlap": sum(1 for row in comparison_rows if row["overlap_found"]),
        "detections_with_correct_class_overlap": sum(1 for row in comparison_rows if row["class_match"]),
    }
    return comparison_rows, summary


def compute_greedy_metrics(
    detection_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    min_iou: float,
) -> dict[str, object]:
    gt_by_video: dict[str, list[dict[str, str]]] = {}
    for row in gt_rows:
        gt_by_video.setdefault(row["video_id"], []).append(row)

    candidates: list[tuple[float, int, str]] = []
    for det_idx, det in enumerate(detection_rows):
        video_id = det["video_id"]
        det_start = to_int(det["start_sample"], "start_sample")
        det_stop = to_int(det["stop_sample"], "stop_sample")
        det_class_id = to_int(det["class_id"], "class_id")

        for gt in gt_by_video.get(video_id, []):
            gt_start = to_int(gt["start_sample"], "start_sample")
            gt_stop = to_int(gt["stop_sample"], "stop_sample")
            gt_class_id = to_int(gt["class_id"], "class_id")
            if det_class_id != gt_class_id:
                continue
            _, _, _, iou = interval_metrics(det_start, det_stop, gt_start, gt_stop)
            if iou >= min_iou:
                candidates.append((iou, det_idx, gt["annotation_id"]))

    candidates.sort(reverse=True)
    used_dets: set[int] = set()
    used_gts: set[str] = set()
    true_positives = 0
    for iou, det_idx, gt_id in candidates:
        if det_idx in used_dets or gt_id in used_gts:
            continue
        used_dets.add(det_idx)
        used_gts.add(gt_id)
        true_positives += 1

    false_positives = len(detection_rows) - true_positives
    false_negatives = len(gt_rows) - true_positives
    precision = 0.0 if len(detection_rows) == 0 else true_positives / len(detection_rows)
    recall = 0.0 if len(gt_rows) == 0 else true_positives / len(gt_rows)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    return {
        "min_iou": min_iou,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def main() -> None:
    args = parse_args()
    detection_rows = read_csv_rows(args.detections_csv)
    gt_rows_all = read_csv_rows(args.ground_truth_csv)
    relevant_video_ids = {row["video_id"] for row in detection_rows}
    gt_rows = [row for row in gt_rows_all if row["video_id"] in relevant_video_ids]

    comparison_rows, summary = compare_rows(detection_rows, gt_rows)
    summary["greedy_class_aware_metrics"] = compute_greedy_metrics(
        detection_rows,
        gt_rows,
        args.min_iou,
    )

    ensure_parent_dir(args.output_csv)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(json.dumps(summary, indent=2))

    if args.summary_json is not None:
        ensure_parent_dir(args.summary_json)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
