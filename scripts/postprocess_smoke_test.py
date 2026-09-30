"""Exercise filtering, reconnection, clustering, and parameter extraction."""

from pathlib import Path
import os
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Avoid one-time Numba compilation during this tiny correctness check. The full
# pipeline uses the default JIT-enabled path for performance.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from step.pic_clustering import cluster_image
from step.pic_eliminate import eliminate_image
from step.pic_extension import extend_image


def main():
    # Two separated, nearly collinear filled-fracture traces provide a compact
    # deterministic fixture without reading a 20,000-row archived interval.
    recognition = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.line(recognition, (20, 180), (112, 132), (0, 0, 128), 5)
    cv2.line(recognition, (142, 116), (235, 68), (0, 0, 128), 5)
    original = np.full_like(recognition, 72)
    print("Filtering synthetic fixture", flush=True)
    filtered = eliminate_image(recognition)
    print("Reconnecting synthetic fixture", flush=True)
    extended, endpoints = extend_image(filtered, reference_point_idx=10, num_expand_points=5)
    print("Clustering synthetic fixture", flush=True)
    result = cluster_image(
        extended,
        original,
        length_m=1.0,
        start_depth_m=0.0,
        diameter_mm=76.0,
        pruning_times=1,
        branch_length_threshold=20,
        clustering_mean_threshold=1.7,
        clustering_std_threshold=1.5,
    )
    expected = {"fracture", "info", "binary", "branches", "cluster_count"}
    if set(result) != expected:
        raise AssertionError(f"Unexpected result keys: {set(result)}")
    print(
        "Post-processing smoke test passed: "
        f"{len(endpoints)} endpoints, {result['cluster_count']} clusters, "
        f"{len(result['fracture'])} fitted fractures"
    )


if __name__ == "__main__":
    main()
