import json
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import call, patch

import numpy as np

from controllers.BeaconBleSignalCrud import normalizeMac
from services.DataProcessService import (
    buildCaptureFingerprints,
    getDeepLearningRefPointsMatrix,
    processBeaconMacs,
    taskToJson,
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

    def test_mac_normalization_is_case_and_separator_independent(self):
        self.assertEqual(
            normalizeMac(' DC-D7-D7-DD-EC-4A '),
            'dc:d7:d7:dd:ec:4a'
        )

    @patch('services.DataProcessService.getObservedMacsByCampaignAndProtocol')
    def test_process_beacon_macs_intersects_observed_signals(self, get_macs):
        get_macs.side_effect = lambda campaign_id, protocol: {
            10: ['aa:aa', 'bb:bb', 'cc:cc'],
            20: ['bb:bb', 'cc:cc', 'dd:dd'],
        }[campaign_id]

        result = processBeaconMacs(10, 20, 'iBeacon')

        self.assertEqual(result, ['bb:bb', 'cc:cc'])
        get_macs.assert_has_calls([
            call(20, 'iBeacon'),
            call(10, 'iBeacon'),
        ])

    def test_task_description_records_the_exact_beacon_inventory(self):
        beacon_macs = ['aa:aa', 'bb:bb']

        task = json.loads(taskToJson(
            'WKNN', 'iBeacon', 37, 20, 'auto', 'manhattan',
            '[1, 5]', 0, [None], [None], [None], [None], beacon_macs
        ))

        self.assertEqual(task['beaconCount'], 2)
        self.assertEqual(task['beaconMacs'], beacon_macs)


if __name__ == '__main__':
    unittest.main()
