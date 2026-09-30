"""Load the published weights on CPU and infer one small video crop."""

from pathlib import Path
import sys
import warnings

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step.pic_recognize import create_recognizer, recognize


def main():
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    video = ROOT / "data" / "example" / "SFGC-BX-007-III-1-9_image.avi"
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not read the first frame of {video}")
    crop = frame[:128, :128]
    model = create_recognizer(device="cpu")
    result = recognize(crop, model, segment_length=128)
    if result.shape != crop.shape:
        raise AssertionError(f"Unexpected result shape: {result.shape}")
    print(f"Model smoke test passed: {crop.shape} -> {result.shape}")


if __name__ == "__main__":
    main()
