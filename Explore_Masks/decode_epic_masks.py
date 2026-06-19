from __future__ import annotations

import argparse
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "EPIC-KITCHENS"


@dataclass
class MaskInstance:
    bbox_norm: tuple[float, float, float, float]
    score: float | None
    class_id: int | None
    counts: str


@dataclass
class HandObjectInstance:
    bbox_norm: tuple[float, float, float, float]
    values: list[float]


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7


def parse_protobuf(data: bytes) -> list[tuple[int, int, Any]]:
    fields: list[tuple[int, int, Any]] = []
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if wire_type == 0:
            value, offset = read_varint(data, offset)
        elif wire_type == 1:
            value = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
        elif wire_type == 2:
            length, offset = read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = struct.unpack_from("<f", data, offset)[0]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        fields.append((field_number, wire_type, value))
    return fields


def parse_bbox_message(data: bytes) -> tuple[float, float, float, float]:
    values_by_field: dict[int, float] = {}
    ordered_values: list[float] = []
    for field_number, wire_type, value in parse_protobuf(data):
        if wire_type == 5:
            ordered_values.append(float(value))
            values_by_field[field_number] = float(value)

    if 3 in values_by_field and 4 in values_by_field:
        # The EPIC pickles omit x1/y1 when the box touches the left/top boundary.
        return (
            values_by_field.get(1, 0.0),
            values_by_field.get(2, 0.0),
            values_by_field[3],
            values_by_field[4],
        )

    if len(ordered_values) >= 4:
        x_values = sorted((ordered_values[0], ordered_values[2]))
        y_values = sorted((ordered_values[1], ordered_values[3]))
        return (x_values[0], y_values[0], x_values[1], y_values[1])

    raise ValueError("Could not recover four bbox coordinates from protobuf message")


def parse_mask_instance(data: bytes) -> MaskInstance:
    bbox_norm: tuple[float, float, float, float] | None = None
    score: float | None = None
    class_id: int | None = None
    counts = ""

    for field_number, wire_type, value in parse_protobuf(data):
        if field_number == 1 and wire_type == 2:
            bbox_norm = parse_bbox_message(value)
        elif field_number == 2 and wire_type == 2:
            counts = value.decode("ascii", errors="replace")
        elif field_number == 3 and wire_type == 5:
            score = float(value)
        elif field_number == 4 and wire_type == 0:
            class_id = int(value)

    if bbox_norm is None:
        raise ValueError("Mask instance did not contain a bounding box")

    return MaskInstance(bbox_norm=bbox_norm, score=score, class_id=class_id, counts=counts)


def parse_hand_object_instance(data: bytes) -> HandObjectInstance:
    fixed_values: list[float] = []
    bbox_norm: tuple[float, float, float, float] | None = None

    for field_number, wire_type, value in parse_protobuf(data):
        if wire_type == 2:
            try:
                candidate_bbox = parse_bbox_message(value)
                if bbox_norm is None:
                    bbox_norm = candidate_bbox
            except Exception:
                for _nested_field_number, nested_wire_type, nested_value in parse_protobuf(value):
                    if nested_wire_type == 5:
                        fixed_values.append(float(nested_value))
        elif wire_type == 5:
            fixed_values.append(float(value))

    if bbox_norm is None and len(fixed_values) >= 4:
        x_values = sorted((fixed_values[0], fixed_values[2]))
        y_values = sorted((fixed_values[1], fixed_values[3]))
        bbox_norm = (x_values[0], y_values[0], x_values[1], y_values[1])

    if bbox_norm is None:
        raise ValueError("Hand-object instance did not contain a usable box")

    return HandObjectInstance(bbox_norm=bbox_norm, values=fixed_values)


def parse_mask_frame(data: bytes) -> tuple[int | None, list[MaskInstance]]:
    frame_index: int | None = None
    instances: list[MaskInstance] = []
    for field_number, wire_type, value in parse_protobuf(data):
        if field_number == 2 and wire_type == 0:
            frame_index = int(value)
        elif field_number == 3 and wire_type == 2:
            instances.append(parse_mask_instance(value))
    return frame_index, instances


def parse_hand_object_frame(data: bytes) -> tuple[str | None, int | None, list[HandObjectInstance]]:
    video_id: str | None = None
    frame_index: int | None = None
    instances: list[HandObjectInstance] = []
    for field_number, wire_type, value in parse_protobuf(data):
        if field_number == 1 and wire_type == 2:
            video_id = value.decode("utf-8", errors="replace")
        elif field_number == 2 and wire_type == 0:
            frame_index = int(value)
        elif wire_type == 2 and field_number >= 3:
            try:
                instances.append(parse_hand_object_instance(value))
            except Exception:
                continue
    return video_id, frame_index, instances


def decode_coco_rle_counts(counts: str) -> list[int]:
    decoded: list[int] = []
    index = 0
    while index < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << (5 * shift)
            more = (char_value & 0x20) != 0
            shift += 1
            if not more and (char_value & 0x10):
                value |= -1 << (5 * shift)
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(int(value))
    return decoded


def decode_rle_mask(counts: str, height: int, width: int) -> np.ndarray:
    run_lengths = decode_coco_rle_counts(counts)
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0
    for run_length in run_lengths:
        if run_length < 0:
            raise ValueError(f"Negative run length encountered: {run_length}")
        end = min(cursor + run_length, total)
        if value == 1 and end > cursor:
            flat[cursor:end] = 1
        cursor = end
        value = 1 - value
    return flat.reshape((height, width), order="F")


def normalized_bbox_to_pixels(
    bbox_norm: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_norm
    left = int(round(x1 * frame_width))
    top = int(round(y1 * frame_height))
    right = int(round(x2 * frame_width))
    bottom = int(round(y2 * frame_height))
    left = max(0, min(frame_width - 1, left))
    top = max(0, min(frame_height - 1, top))
    right = max(left + 1, min(frame_width, right))
    bottom = max(top + 1, min(frame_height, bottom))
    return left, top, right, bottom


def pick_mask_shape(instance: MaskInstance, frame_width: int, frame_height: int) -> tuple[int, int]:
    total_pixels = sum(decode_coco_rle_counts(instance.counts))
    if total_pixels == frame_width * frame_height:
        return frame_height, frame_width

    root = int(round(total_pixels ** 0.5))
    if root * root == total_pixels:
        return root, root

    left, top, right, bottom = normalized_bbox_to_pixels(instance.bbox_norm, frame_width, frame_height)
    candidates = [
        (bottom - top, right - left),
        (bottom - top + 1, right - left),
        (bottom - top, right - left + 1),
        (bottom - top + 1, right - left + 1),
    ]
    for height, width in candidates:
        if height > 0 and width > 0 and height * width == total_pixels:
            return height, width

    for width in range(1, int(np.sqrt(total_pixels)) + 2):
        if total_pixels % width == 0:
            height = total_pixels // width
            if abs(width - (right - left)) <= 2 and abs(height - (bottom - top)) <= 2:
                return height, width

    raise ValueError(
        "Could not infer mask raster shape "
        f"for bbox={instance.bbox_norm} total_pixels={total_pixels}"
    )


def place_mask_on_canvas(
    instance: MaskInstance,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    mask_height, mask_width = pick_mask_shape(instance, frame_width, frame_height)
    decoded = decode_rle_mask(instance.counts, mask_height, mask_width)

    if mask_height != frame_height or mask_width != frame_width:
        decoded = cv2.resize(decoded, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    return decoded.astype(np.uint8)


def summarize_pickles(mask_records: list[bytes], hand_records: list[bytes], sample_frames: int) -> None:
    non_empty_masks = 0
    non_empty_hand_objects = 0
    print(f"mask_records={len(mask_records)} hand_records={len(hand_records)}")

    for blob in mask_records:
        _frame_index, instances = parse_mask_frame(blob)
        if instances:
            non_empty_masks += 1

    for blob in hand_records:
        _video_id, _frame_index, instances = parse_hand_object_frame(blob)
        if instances:
            non_empty_hand_objects += 1

    print(f"frames_with_masks={non_empty_masks}")
    print(f"frames_with_hand_objects={non_empty_hand_objects}")

    shown = 0
    for frame_offset, blob in enumerate(mask_records):
        frame_index, instances = parse_mask_frame(blob)
        if not instances:
            continue
        print(f"\nframe_offset={frame_offset} frame_index={frame_index} mask_instances={len(instances)}")
        for idx, instance in enumerate(instances[:3]):
            print(
                "  "
                f"mask[{idx}] bbox_norm={tuple(round(v, 4) for v in instance.bbox_norm)} "
                f"score={None if instance.score is None else round(instance.score, 4)} "
                f"class_id={instance.class_id} "
                f"rle_chars={len(instance.counts)}"
            )
        shown += 1
        if shown >= sample_frames:
            break


def overlay_instances(
    frame: np.ndarray,
    mask_instances: list[MaskInstance],
    hand_instances: list[HandObjectInstance],
    alpha: float,
) -> np.ndarray:
    output = frame.copy()
    overlay = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    colors = [
        (0, 255, 255),
        (0, 220, 120),
        (255, 180, 0),
        (255, 0, 180),
        (80, 160, 255),
    ]

    for idx, instance in enumerate(mask_instances):
        color = colors[idx % len(colors)]
        try:
            mask = place_mask_on_canvas(instance, frame_width, frame_height)
        except Exception as exc:
            print(f"Skipping mask {idx}: {exc}")
            continue

        overlay[mask > 0] = (
            alpha * np.array(color, dtype=np.float32) + (1.0 - alpha) * overlay[mask > 0]
        ).astype(np.uint8)

        left, top, right, bottom = normalized_bbox_to_pixels(instance.bbox_norm, frame_width, frame_height)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        label = f"mask {idx}"
        if instance.score is not None:
            label += f" {instance.score:.2f}"
        cv2.putText(
            output,
            label,
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)

    for idx, instance in enumerate(hand_instances):
        color = (255, 255, 255)
        left, top, right, bottom = normalized_bbox_to_pixels(instance.bbox_norm, frame_width, frame_height)
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.putText(
            output,
            f"hand/object {idx}",
            (left, min(frame_height - 10, bottom + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def render_video(
    video_path: Path,
    mask_records: list[bytes],
    hand_records: list[bytes],
    output_path: Path,
    start_frame: int,
    num_frames: int,
    alpha: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    end_frame = min(start_frame + num_frames, total_frames, len(mask_records), len(hand_records))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break

        mask_frame_index, mask_instances = parse_mask_frame(mask_records[frame_idx])
        _video_id, hand_frame_index, hand_instances = parse_hand_object_frame(hand_records[frame_idx])

        annotated = overlay_instances(frame, mask_instances, hand_instances, alpha=alpha)
        cv2.putText(
            annotated,
            f"video_frame={frame_idx} mask_frame={mask_frame_index} hand_frame={hand_frame_index}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"video_frame={frame_idx} mask_frame={mask_frame_index} hand_frame={hand_frame_index}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(annotated)

    writer.release()
    cap.release()
    print(f"Saved overlay video to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode EPIC-KITCHENS mask and hand-object pickle files.")
    parser.add_argument(
        "--mask-pkl",
        type=Path,
        default=DEFAULT_DATA_ROOT / "P01" / "masks" / "P01_01.pkl",
    )
    parser.add_argument(
        "--hand-pkl",
        type=Path,
        default=DEFAULT_DATA_ROOT / "P01" / "hand-objects" / "P01_01.pkl",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_DATA_ROOT / "P01" / "videos" / "P01_01.MP4",
    )
    parser.add_argument("--mode", choices=["summary", "render"], default="summary")
    parser.add_argument("--sample-frames", type=int, default=5)
    parser.add_argument("--start-frame", type=int, default=180)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Explore_Masks" / "outputs" / "P01_01_masks_overlay.mp4",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_records = pickle.load(args.mask_pkl.open("rb"))
    hand_records = pickle.load(args.hand_pkl.open("rb"))

    if args.mode == "summary":
        summarize_pickles(mask_records, hand_records, sample_frames=args.sample_frames)
        return

    render_video(
        video_path=args.video,
        mask_records=mask_records,
        hand_records=hand_records,
        output_path=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
