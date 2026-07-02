import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from services.DataProcessService import (
    buildCaptureFingerprints,
    getDeepLearningRefPointsMatrix,
)
from services.MethodsService import MISSING_RSSI


@dataclass
class Sample:
    Date: datetime
    RSSI: float


class DataProcessServiceTest(unittest.TestCase):
    @patch('services.DataProcessService.getFilteredPredataSamples')
    def test_deep_learning_uses_only_aleatory_rotations(self, get_samples):
        get_samples.return_value = [
            Sample(datetime(2026, 4, 10, 10), -60)
        ]
        points = [{'x': 0.0, 'y': 0.0, 'z': 0.0}]
        configuration = {
            'POINT_0': {
                'X': '0', 'Y': '0', 'Z': '0',
                'Rotations': [0.0, 90.0],
            }
        }

        matrix = getDeepLearningRefPointsMatrix(
            points, configuration, ['beacon'], 1,
            'iBeacon', 37, 0, {0.0}
        )

        self.assertEqual(matrix.shape, (7, 4))
        self.assertTrue(all(
            call.args[2] == 0.0 for call in get_samples.call_args_list
        ))

    def test_capture_fingerprints_preserve_dates_and_apply_sample_limit(self):
        first = datetime(2026, 4, 10, 10)
        second = first + timedelta(days=1)
        samples_by_beacon = [
            [
                Sample(first, -70),
                Sample(first, -60),
                Sample(second, -80),
                Sample(second, -75),
            ],
            [
                Sample(first, -55),
                Sample(second, -65),
            ],
        ]

        fingerprints = buildCaptureFingerprints(
            samples_by_beacon,
            rssiSamples=1,
            minimumObservedBeacons=1,
        )

        np.testing.assert_array_equal(fingerprints[0], [-70, -55])
        np.testing.assert_array_equal(fingerprints[1], [-80, -65])
        np.testing.assert_array_equal(fingerprints[2], [-70, -55])

    def test_sparse_captures_are_removed_but_aggregate_is_kept(self):
        first = datetime(2026, 4, 10, 10)
        second = first + timedelta(days=1)
        samples_by_beacon = [
            [Sample(first, -60)],
            [Sample(second, -70)],
            [Sample(second, -80)],
        ]

        fingerprints = buildCaptureFingerprints(
            samples_by_beacon,
            rssiSamples=0,
            minimumObservedBeacons=2,
        )

        self.assertEqual(len(fingerprints), 2)
        np.testing.assert_array_equal(
            fingerprints[0], [MISSING_RSSI, -70, -80]
        )
        np.testing.assert_array_equal(fingerprints[1], [-60, -70, -80])


if __name__ == '__main__':
    unittest.main()
