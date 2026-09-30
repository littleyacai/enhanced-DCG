"""Validate the inventory and basic scientific structure of archived outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reference_outputs" / "full_example"
INTERVALS = [
    "0m_to_5m", "5m_to_10m", "10m_to_15m", "15m_to_20m",
    "20m_to_25m", "25m_to_30m", "30m_to_35m", "35m_to_40m",
    "40m_to_45m", "45m_to_46.3m",
]


def load_image(path: Path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path.relative_to(ROOT)}")
    return image


def main() -> int:
    failures = []
    counts = {}
    for interval in INTERVALS:
        names = {
            "borehole_segments": f"segment_{interval}.png",
            "recognize": f"segment_{interval}.png",
            "eliminate": f"segment_{interval}.png",
            "extension": f"segment_{interval}.png",
            "result": f"segment_{interval}-image2D.png",
        }
        images = {}
        for stage, name in names.items():
            path = ARCHIVE / stage / name
            if not path.is_file():
                failures.append(f"missing {path.relative_to(ROOT)}")
                continue
            try:
                images[stage] = load_image(path)
            except ValueError as exc:
                failures.append(str(exc))
        if len(images) == 5:
            shapes = {image.shape[:2] for image in images.values()}
            if len(shapes) != 1:
                failures.append(f"stage dimensions disagree for {interval}: {shapes}")
            counts[interval] = {
                stage: int(np.count_nonzero(np.any(image != 0, axis=2)))
                for stage, image in images.items() if stage in {"recognize", "eliminate", "extension"}
            }
            if counts[interval]["eliminate"] > counts[interval]["recognize"]:
                failures.append(f"filtering added foreground pixels for {interval}")

        workbook = ARCHIVE / "result" / f"segment_{interval}-fracture_parameters.xlsx"
        if not workbook.is_file():
            failures.append(f"missing {workbook.relative_to(ROOT)}")
        else:
            try:
                table = pd.read_excel(workbook)
                required = {"深度(m)", "倾向(°)", "倾角(°)"}
                if not required.issubset(table.columns):
                    failures.append(f"unexpected workbook columns: {workbook.name}")
                if len(table) and not table["倾角(°)"].between(0, 90).all():
                    failures.append(f"dip angle outside 0–90 degrees: {workbook.name}")
            except Exception as exc:
                failures.append(f"cannot read {workbook.name}: {exc}")

    print(f"Validated {len(INTERVALS)} depth intervals under {ARCHIVE.relative_to(ROOT)}")
    for interval, values in counts.items():
        print(interval, values)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("Reference-output validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
