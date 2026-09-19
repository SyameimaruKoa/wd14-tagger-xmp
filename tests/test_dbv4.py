import csv
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from dbv4 import (
    DBV4Metadata,
    DBV4Preprocessor,
    adapt_input_layout,
    detect_input_layout,
    infer_output_to_probabilities,
)


class DBV4MetadataTests(unittest.TestCase):
    def _make_metadata(self, directory):
        with open(os.path.join(directory, "model.onnx"), "wb") as f:
            f.write(b"placeholder")
        with open(os.path.join(directory, "selected_tags.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "category", "best_threshold"])
            writer.writerow(["general", 9, 0.21])
            writer.writerow(["sensitive", 9, 0.22])
            writer.writerow(["questionable", 9, 0.23])
            writer.writerow(["explicit", 9, 0.24])
            writer.writerow(["tag_a", 0, 0.31])
        with open(os.path.join(directory, "categories.json"), "w", encoding="utf-8") as f:
            json.dump([
                {"category": 0, "name": "general"},
                {"category": 4, "name": "character"},
                {"category": 9, "name": "rating"},
            ], f)
        with open(os.path.join(directory, "thresholds.csv"), "w", encoding="utf-8") as f:
            f.write("category,name,alpha,threshold\n0,general,1,0.35\n")
        with open(os.path.join(directory, "preprocess.json"), "w", encoding="utf-8") as f:
            json.dump({
                "test": [
                    {"type": "PadToSize", "params": {"size": [8, 8], "background_color": "white"}},
                    {"type": "Resize", "params": {"size": 8, "interpolation": "bicubic"}},
                    {"type": "CenterCrop", "params": {"size": [8, 8]}},
                    {"type": "MaybeToTensor"},
                    {"type": "Normalize", "params": {
                        "mean": [0.5, 0.5, 0.5],
                        "std": [0.5, 0.5, 0.5],
                    }},
                ],
            }, f)
        profile = {
            "profile_name": "test",
            "repo_id": "",
            "model_file": "model.onnx",
            "tags_file": "selected_tags.csv",
            "preprocess_file": "preprocess.json",
            "categories_file": "categories.json",
            "thresholds_file": "thresholds.csv",
        }
        return DBV4Metadata.load(profile, base_dir=directory)

    def test_metadata_and_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = self._make_metadata(directory)
            self.assertEqual(metadata.label_count, 5)
            self.assertEqual(metadata.threshold_for("tag_a"), 0.31)
            self.assertEqual(metadata.rating_indices["explicit"], 3)
            self.assertEqual(metadata.rating_tags_marker, "dbv4_model:")
            self.assertEqual(len(metadata.metadata_version), 16)

    def test_preprocess_and_layout(self):
        preprocessor = DBV4Preprocessor({
            "test": [
                {"type": "PadToSize", "params": {"size": [8, 8], "background_color": "white"}},
                {"type": "Resize", "params": {"size": 8, "interpolation": "bicubic"}},
                {"type": "CenterCrop", "params": {"size": [8, 8]}},
                {"type": "MaybeToTensor"},
                {"type": "Normalize", "params": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                }},
            ]
        })
        value = preprocessor(Image.new("RGB", (2, 4), (255, 0, 0)))
        self.assertEqual(value.shape, (3, 8, 8))
        self.assertTrue(np.isfinite(value).all())
        self.assertEqual(detect_input_layout([None, 3, 8, 8]), "NCHW")
        self.assertEqual(detect_input_layout([None, 8, 8, 3]), "NHWC")
        batch = value[None, ...]
        self.assertEqual(adapt_input_layout(batch, [None, 3, 8, 8]).shape, (1, 3, 8, 8))
        self.assertEqual(adapt_input_layout(batch, [None, 8, 8, 3]).shape, (1, 8, 8, 3))

    def test_output_probability_normalization(self):
        probabilities = infer_output_to_probabilities(np.array([0.0, 2.0, -2.0], dtype=np.float32))
        np.testing.assert_allclose(probabilities, [0.5, 0.8807971, 0.11920292], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
