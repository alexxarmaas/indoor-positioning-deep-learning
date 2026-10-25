import unittest

import numpy as np

from services.TFGFeaturesService import (
    BEACONS,
    MISSING_RSSI,
    PRESENCE_COLUMNS,
    RSSI_COLUMNS,
    build_tfg_fingerprint,
    load_beacon_mapping,
)


class TestTFGFeaturesService(unittest.TestCase):
    def test_mapping_is_complete_and_ordered(self):
        mapping = load_beacon_mapping()
        self.assertEqual(list(mapping), BEACONS)
        self.assertEqual(len(set(mapping.values())), 8)
        self.assertEqual(len(RSSI_COLUMNS), 8)
        self.assertEqual(len(PRESENCE_COLUMNS), 8)

    def test_fingerprint_conversion(self):
        mapping = load_beacon_mapping()
        signals = [
            {"Mac": mapping["b01"].upper(), "RSSI": -60},
            {"Mac": mapping["b01"], "RSSI": -70},
            {"Mac": mapping["b03"], "RSSI": -82},
            {"Mac": "00:00:00:00:00:00", "RSSI": -20},
        ]
        result = build_tfg_fingerprint(signals, mapping)
        self.assertEqual(result.shape, (1, 16))
        self.assertEqual(result[0, 0], -65.0)
        self.assertEqual(result[0, 8], 1.0)
        self.assertEqual(result[0, 1], MISSING_RSSI)
        self.assertEqual(result[0, 9], 0.0)
        self.assertEqual(result[0, 2], -82.0)
        self.assertEqual(result[0, 10], 1.0)
        np.testing.assert_array_equal(result[0, 3:8], [MISSING_RSSI] * 5)
        np.testing.assert_array_equal(result[0, 11:16], [0.0] * 5)

    def test_observed_negative_100_is_distinct_from_missing(self):
        mapping = load_beacon_mapping()
        fingerprint = build_tfg_fingerprint([
            {"Mac": mapping["b01"], "RSSI": -100}
        ], mapping)
        self.assertEqual(fingerprint[0, 0], -100.0)
        self.assertEqual(fingerprint[0, 8], 1.0)
        self.assertEqual(fingerprint[0, 1], -100.0)
        self.assertEqual(fingerprint[0, 9], 0.0)


if __name__ == "__main__":
    unittest.main()
