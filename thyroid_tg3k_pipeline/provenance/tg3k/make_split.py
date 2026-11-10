"""Create a leakage-reduced, within-block TG3K train/validation split.

The blocks are inferred from contiguous runs of image dimensions. They are
observable groups, not verified patient or acquisition identifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def dimension(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(r"E:\Thyroid Dataset\tg3k"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    image_dir = args.data / "thyroid-image"
    mask_dir = args.data / "thyroid-mask"
    ids = sorted(int(path.stem) for path in image_dir.glob("*.jpg"))
    if ids != list(range(len(ids))):
        raise ValueError("Image IDs must be contiguous, starting at zero")
    if {path.name for path in image_dir.glob("*.jpg")} != {
        path.name for path in mask_dir.glob("*.jpg")
    }:
        raise ValueError("Image and mask filenames do not match")

    sizes = [dimension(image_dir / f"{index:04d}.jpg") for index in ids]
    if any(
        dimension(mask_dir / f"{index:04d}.jpg") != sizes[index]
        for index in ids
    ):
        raise ValueError("An image and its mask have different dimensions")

    runs = []
    start = 0
    for index in range(1, len(ids) + 1):
        if index == len(ids) or sizes[index] != sizes[start]:
            runs.append((start, index - 1, sizes[start]))
            start = index

    rows = []
    groups = []
    split_ids = {"train": [], "val": [], "unused": []}
    for group_number, (start, end, size) in enumerate(runs, 1):
        length = end - start + 1
        val_length = round(0.15 * length)
        center_fraction = (0.30, 0.50, 0.70)[(group_number - 1) % 3]
        center = start + round(center_fraction * (length - 1))
        val_start = center - val_length // 2
        val_end = val_start + val_length - 1
        gap = max(25, round(0.10 * length))
        group_name = f"G{group_number:02d}"

        counts = {"train": 0, "val": 0, "unused": 0}
        for index in range(start, end + 1):
            if val_start <= index <= val_end:
                split = "val"
            elif val_start - gap <= index < val_start or val_end < index <= val_end + gap:
                split = "unused"
            else:
                split = "train"
            counts[split] += 1
            split_ids[split].append(index)
            rows.append(
                {
                    "id": index,
                    "group": group_name,
                    "split": split,
                    "image": f"thyroid-image/{index:04d}.jpg",
                    "mask": f"thyroid-mask/{index:04d}.jpg",
                }
            )

        if not counts["train"] or not counts["val"]:
            raise ValueError(f"{group_name} does not appear in both splits")
        groups.append(
            {
                "group": group_name,
                "start": start,
                "end": end,
                "width": size[0],
                "height": size[1],
                "val_start": val_start,
                "val_end": val_end,
                "gap_each_side": gap,
                **counts,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    for filename, data, fieldnames in [
        ("split_manifest.csv", rows, ["id", "group", "split", "image", "mask"]),
        (
            "group_summary.csv",
            groups,
            [
                "group", "start", "end", "width", "height", "val_start",
                "val_end", "gap_each_side", "train", "val", "unused",
            ],
        ),
    ]:
        with (args.out / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    (args.out / "split_ids.json").write_text(
        json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"groups={len(groups)} images={len(ids)} "
        + " ".join(f"{key}={len(value)}" for key, value in split_ids.items())
    )


if __name__ == "__main__":
    main()
