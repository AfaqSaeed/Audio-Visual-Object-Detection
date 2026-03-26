import csv
from pathlib import Path


def find_first_video(video_root: Path) -> Path:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted([p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not video_files:
        raise FileNotFoundError(f"No video files found under: {video_root}")
    return video_files[0]


def find_wav_for_video(sound_root: Path, video_id: str) -> Path | None:
    # Prefer exact match: <video_id>.wav
    exact = sound_root.rglob(f"{video_id}.wav")
    for p in exact:
        if p.is_file():
            return p
    # Fallback: any wav containing the video_id prefix
    candidates = sorted([p for p in sound_root.rglob("*.wav") if p.is_file() and p.stem == video_id])
    return candidates[0] if candidates else None


def read_filtered_rows(csv_path: Path, video_id: str) -> list[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("video_id") == video_id]
    return rows


def format_row(r: dict) -> str:
    return (
        f"{r.get('annotation_id')} | participant={r.get('participant_id')} | video_id={r.get('video_id')} | "
        f"{r.get('start_timestamp')}-{r.get('stop_timestamp')} | desc={r.get('description')} | "
        f"class={r.get('class')} (class_id={r.get('class_id')})"
    )


def main() -> None:
    # Match your dataset locations
    video_root = Path(r"D:\datasets\EPIC-KITCHENS")
    sound_root = Path(r"D:\datasets\sound files")
    train_csv = Path(r"D:\Audio-Visual Object Detection\epic-sounds-annotations\EPIC_Sounds_train.csv")
    val_csv = Path(r"D:\Audio-Visual Object Detection\epic-sounds-annotations\EPIC_Sounds_validation.csv")

    # Pick one video (first found, sorted) and derive video_id from filename stem.
    video_path = find_first_video(video_root)
    video_id = video_path.stem

    wav_path = find_wav_for_video(sound_root, video_id)

    train_rows = read_filtered_rows(train_csv, video_id)
    val_rows = read_filtered_rows(val_csv, video_id)

    first_n = 10

    print(f"Sample video: {video_path}")
    print(f"Derived video_id: {video_id}")
    print(f"Matching wav: {wav_path if wav_path else 'NOT FOUND'}")
    print("")

    print(f"Rows where video_id == {video_id}")
    print(f"- train CSV: {len(train_rows)} matches")
    print(f"- validation CSV: {len(val_rows)} matches")
    print("")

    print(f"First {first_n} train rows:")
    for r in train_rows[:first_n]:
        print(format_row(r))
    if not train_rows:
        print("(no matching train rows)")
    print("")

    print(f"First {first_n} validation rows:")
    for r in val_rows[:first_n]:
        print(format_row(r))
    if not val_rows:
        print("(no matching validation rows)")


if __name__ == "__main__":
    main()

