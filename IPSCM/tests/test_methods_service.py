import json
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from services.MethodsService import (
    applyMethod,
    calculateError,
    getDeepLearningConfiguration,
    getKeyDescription,
    getPyTorchDeepLearningConfiguration,
    inverseDistanceWeights,
    prepareDeepLearningFeatures,
    prepareRSSIFeatures,
    validateFiniteResults,
)


class MethodsServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        coordinates = np.array(
            [[x, y, 0.0] for x in range(6) for y in range(5)],
            dtype=float,
        )
        cls.train = np.column_stack((cls._features(coordinates), coordinates))

        test_coordinates = np.array([
            [0.5, 0.5, 0.0],
            [1.5, 2.5, 0.0],
            [3.5, 1.5, 0.0],
            [4.5, 3.5, 0.0],
        ])
        cls.test = np.column_stack((cls._features(test_coordinates), test_coordinates))
        cls.train[0, 1] = -np.inf
        cls.test[1, 1] = -np.inf

    @staticmethod
    def _features(coordinates):
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        return np.column_stack((
            -40.0 - 3.0 * x - y,
            -50.0 + x - 2.0 * y,
            -60.0 - (x - y) ** 2,
        ))

    @staticmethod
    def _description(method, metric=0, algorithm=0):
        description = {
            'method': method,
            'protocol': 'iBeacon',
            'channel': 37,
            'rssiSample': 10,
            'algorithm': algorithm,
            'metric': metric,
            'ksRange': '[1, 3]',
            'kernel': 0,
            'cs': [None],
            'gammas': [None],
            'nus': [None],
            'Is': [None],
        }
        if method == 'DeepLearning':
            description['deepLearning'] = getDeepLearningConfiguration()
        elif method == 'DeepLearningPyTorch':
            description['deepLearningPyTorch'] = (
                getPyTorchDeepLearningConfiguration()
            )
        return json.dumps(description)

    def test_deep_learning_predictions_are_finite_and_reproducible(self):
        description = self._description('DeepLearning')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            first = applyMethod(
                self.train, self.test, 'DeepLearning', 0, 0, [0, 0],
                0, [None], [None], [None], [None], description,
            )
            second = applyMethod(
                self.train, self.test, 'DeepLearning', 0, 0, [0, 0],
                0, [None], [None], [None], [None], description,
            )

        self.assertEqual(len(first), 1)
        key = next(iter(first))
        self.assertIn('[DeepLearning]', key)
        self.assertIn('[layers=', key)
        self.assertIn('[ensemble=6]', key)
        prediction = np.asarray(first[key])
        self.assertEqual(prediction.shape, (len(self.test), 3))
        self.assertTrue(np.isfinite(prediction).all())
        np.testing.assert_allclose(first[key], second[key])

    def test_deep_learning_and_wknn_share_the_error_metric(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            deep_learning = applyMethod(
                self.train, self.test, 'DeepLearning', 0, 0, [0, 0],
                0, [None], [None], [None], [None],
                self._description('DeepLearning'),
            )

        wknn = applyMethod(
            self.train, self.test, 'WKNN', 'manhattan', 'brute', [1, 3],
            0, [None], [None], [None], [None],
            self._description('WKNN', 'manhattan', 'brute'),
        )

        for results in (deep_learning, wknn):
            errors = calculateError(self.test, results)
            json.dumps(errors)
            self.assertEqual(len(errors), len(results))
            for values in errors.values():
                self.assertEqual(len(values), len(self.test))
                self.assertTrue(np.isfinite(values).all())
                self.assertEqual(values, sorted(values))

    def test_missing_rssi_is_replaced_with_the_configured_floor(self):
        features = np.array([[-50.0, np.nan, -np.inf, np.inf]])
        prepared = prepareRSSIFeatures(features)
        np.testing.assert_array_equal(prepared, [[-50.0, -110.0, -110.0, -110.0]])

    def test_deep_learning_features_add_relative_rssi_and_missing_mask(self):
        features = np.array([[-50.0, -60.0, np.nan]])
        prepared = prepareDeepLearningFeatures(features)

        self.assertEqual(prepared.shape, (1, 9))
        np.testing.assert_array_equal(prepared[0, :3], [-50.0, -60.0, -110.0])
        np.testing.assert_array_equal(prepared[0, 3:6], [5.0, -5.0, 0.0])
        np.testing.assert_array_equal(prepared[0, 6:], [0.0, 0.0, 1.0])

    def test_inverse_distance_weights_handle_exact_matches_without_infinity(self):
        distances = np.array([[0.0, 2.0], [1.0, 2.0]])
        weights = inverseDistanceWeights(distances)

        np.testing.assert_array_equal(weights[0], [1.0, 0.0])
        np.testing.assert_array_equal(weights[1], [1.0, 0.5])
        self.assertTrue(np.isfinite(weights).all())

    def test_beacon_inventory_is_hidden_from_visible_series_labels(self):
        description = json.loads(
            self._description('WKNN', 'manhattan', 'brute')
        )
        description['beaconCount'] = 2
        description['beaconMacs'] = ['aa:aa', 'bb:bb']

        label = getKeyDescription(json.dumps(description))

        self.assertIn('[WKNN]', label)
        self.assertNotIn('[2]', label)
        self.assertNotIn('aa:aa', label)

    def test_non_finite_results_are_rejected_before_persistence(self):
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            validateFiniteResults({'prediction': [[1.0, np.nan, 3.0]]}, 'Prediction')


if __name__ == '__main__':
    unittest.main()
