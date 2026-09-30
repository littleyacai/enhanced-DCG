"""Fast regression tests for portable public functions."""

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from step.evaluation import dip2normal, qualityCheck
from step.excel_export import write_workbook
from step.pic_eliminate import eliminate_image


class EvaluationTests(unittest.TestCase):
    def test_identical_fractures_score_one(self):
        fractures = [[1.0, 0.0, 0.0], [2.0, 125.0, 45.0]]
        self.assertEqual(qualityCheck(fractures, fractures, 10.0), (1.0, 1.0, 1.0))

    def test_north_dipping_normal(self):
        np.testing.assert_allclose(dip2normal(0, 90), [0, 1, 0], atol=1e-12)

    def test_empty_prediction_scores_zero(self):
        self.assertEqual(qualityCheck([], [[1.0, 0.0, 20.0]], 10.0), (0.0, 0.0, 0.0))


class FilteringTests(unittest.TestCase):
    def test_small_component_is_removed(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        image[2:4, 2:4] = (0, 128, 0)
        cv2.line(image, (10, 2), (10, 25), (0, 0, 128), 1)
        result = eliminate_image(image, area_threshold=10,
                                 aspect_ratio_threshold=0.5,
                                 rectangularity_threshold=0.5)
        self.assertFalse(np.any(result[2:4, 2:4]))
        self.assertTrue(np.any(result[:, 10]))


class WorkbookTests(unittest.TestCase):
    def test_portable_excel_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.xlsx"
            write_workbook(path, {"Results": pd.DataFrame({"depth": [1.0], "value": [np.nan]})})
            workbook = load_workbook(path, data_only=True)
            self.assertEqual(workbook["Results"]["A2"].value, 1)
            self.assertIsNone(workbook["Results"]["B2"].value)


if __name__ == "__main__":
    unittest.main()
