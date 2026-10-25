import json
import unittest
from unittest.mock import patch

import numpy as np

from services.TFGPositioningService import (
    applyTFGMLPmethod,
    tfg_wknn_matrix,
)


def make_matrix(n, z=1.2):
    rssis = np.tile(np.linspace(-86.0, -49.0, 8), (n, 1))
    rssis[:, 0] += np.arange(n)
    masks = np.ones((n, 8), dtype=float)
    masks[::2, 3] = 0
    rssis[::2, 3] = -100
    coordinates = np.column_stack((
        np.arange(n, dtype=float) % 6,
        np.arange(n, dtype=float) // 6,
        np.full(n, z)
    ))
    return np.column_stack((rssis, masks, coordinates))


class FakeRegressor:
    def fit(self, features, targets):
        self.features_ = np.asarray(features)
        self.targets_ = np.asarray(targets)
        return self

    def predict(self, features):
        return np.zeros((len(features), 2))


class TFGPositioningTest(unittest.TestCase):
    def test_tfg_wknn_reuses_only_eight_rssi_columns(self):
        full = make_matrix(15)
        converted = tfg_wknn_matrix(full)
        self.assertEqual(converted.shape, (15, 11))
        np.testing.assert_array_equal(converted[:, :8], full[:, :8])
        np.testing.assert_array_equal(converted[:, -3:], full[:, -3:])

    @patch("services.TFGPositioningService.MLPRegressor")
    def test_mask_and_robust_variants_preserve_feature_contract(self, model_type):
        instances = []
        def create_model(**kwargs):
            self.assertEqual(kwargs["hidden_layer_sizes"], (64, 32))
            self.assertEqual(kwargs["random_state"], 42)
            instance = FakeRegressor()
            instances.append(instance)
            return instance
        model_type.side_effect = create_model
        train = make_matrix(18)
        test = make_matrix(6)
        for use_mask, width in ((False, 8), (True, 16)):
            description = json.dumps({
                "method": "MLP_TFG_MASK" if use_mask else "MLP_TFG_ROBUST"
            })
            result = applyTFGMLPmethod(train, test, description, use_mask)
            self.assertEqual(len(result), 1)
            prediction = np.asarray(next(iter(result.values())))
            self.assertEqual(prediction.shape, (6, 3))
            np.testing.assert_array_equal(prediction[:, 2], [1.2] * 6)
            self.assertEqual(instances[-1].features_.shape, (18, width))
            if use_mask:
                np.testing.assert_array_equal(instances[-1].features_[:, 8:], train[:, 8:16])

    def test_reject_nonconstant_z(self):
        train = make_matrix(18)
        train[3, -1] = 5.0
        with self.assertRaisesRegex(ValueError, "constante"):
            applyTFGMLPmethod(
                train, make_matrix(6),
                json.dumps({"method": "MLP_TFG_ROBUST"})
            )


if __name__ == "__main__":
    unittest.main()
