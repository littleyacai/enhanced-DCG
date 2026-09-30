"""Combine the archived segment workbooks into portable UTF-8 CSV tables."""

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reference_outputs" / "segment_tables"
DESTINATION = ROOT / "reference_outputs"
COLUMNS = {
    "裂隙编号": "fracture_id",
    "裂隙类型": "fracture_type_label",
    "类型标识": "fracture_type",
    "深度(m)": "depth_m",
    "开度(mm)": "aperture_mm",
    "迹线长度(mm)": "trace_length_mm",
    "开度乘迹线长度(mm²)": "aperture_trace_product_mm2",
    "倾向(°)": "dip_direction_deg",
    "倾角(°)": "dip_angle_deg",
    "法向量X": "normal_x",
    "法向量Y": "normal_y",
    "法向量Z": "normal_z",
    "骨架点数": "skeleton_pixel_count",
}


def interval_start(path):
    match = re.search(r"segment_([0-9.]+)m_to_([0-9.]+)m", path.name)
    if not match:
        raise ValueError(f"Cannot parse interval from {path.name}")
    return float(match.group(1)), float(match.group(2))


def main():
    frames = []
    for path in sorted(SOURCE.glob("*.xlsx"), key=lambda item: interval_start(item)[0]):
        start, end = interval_start(path)
        frame = pd.read_excel(path).rename(columns=COLUMNS)
        missing = set(COLUMNS.values()) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frame.insert(0, "source_workbook", path.name)
        frame.insert(1, "segment_start_m", start)
        frame.insert(2, "segment_end_m", end)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No XLSX files found in {SOURCE}")
    populated = [frame for frame in frames if not frame.empty]
    combined = pd.concat(populated, ignore_index=True) if populated else frames[0].iloc[:0].copy()
    combined.to_csv(DESTINATION / "all_segment_fractures.csv", index=False, encoding="utf-8")
    summary = (
        combined.groupby("fracture_type", dropna=False)
        .agg(
            fracture_count=("fracture_id", "count"),
            mean_aperture_mm=("aperture_mm", "mean"),
            mean_trace_length_mm=("trace_length_mm", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(DESTINATION / "fracture_type_summary.csv", index=False, encoding="utf-8")
    print(f"Combined {len(combined)} fractures from {len(frames)} segment workbooks.")


if __name__ == "__main__":
    main()
