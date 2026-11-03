import unittest
from types import SimpleNamespace
import numpy as np


from services.TFGFeaturesService import (
    load_beacon_mapping,
    build_tfg_fingerprint, 
    build_tfg_dataset,
    RSSI_COLUMNS, 
    PRESENCE_COLUMNS
)

class TestTFGFeaturesService(unittest.TestCase):

    def test_fingerprint_conversion(self):
        mapping = load_beacon_mapping()

        signals = [
            {"Mac": mapping["b01"], "RSSI": -60},
            {"Mac": mapping["b01"], "RSSI": -70},
            {"Mac": mapping["b03"], "RSSI": -82},
        ]

        result = build_tfg_fingerprint(signals, mapping)

        self.assertEqual(result.shape, (1,16))

        self.assertEqual(result[0, 0], -65)
        self.assertEqual(result[0, 8], 1)

        self.assertEqual(result[0, 1], -100)
        self.assertEqual(result[0, 9], 0)

        self.assertEqual(result[0, 2], -82.0)
        self.assertEqual(result[0, 10], 1)

    def test_build_tfg_dataset(self):
        mapping = load_beacon_mapping()

        captures = [
            SimpleNamespace(
                Id = 1,
                Position_x = 1.0,
                Position_y = 2.0,
            ),
            SimpleNamespace(
                Id = 2,
                Position_x = 4.0,
                Position_y = 5.0,
            )
        ]

        signals = [
            SimpleNamespace(
                Id_capture = 1,
                Mac = mapping["b01"],
                RSSI = -60
            ),
            SimpleNamespace(
                Id_capture = 1,
                Mac = mapping["b01"],
                RSSI = -70
            ),
            SimpleNamespace(
                Id_capture = 2,
                Mac = mapping["b02"],
                RSSI = -80
            )
        ]

        x, y, capture_ids = build_tfg_dataset(captures, signals)

        self.assertEqual(x.shape, (2,16))
        self.assertEqual(y.shape, (2,2))

        self.assertEqual(x[0, 0], -65)
        self.assertEqual(x[0, 8], 1)

        self.assertEqual(x[1, 1], -80)
        self.assertEqual(x[1, 9], 1)

        self.assertEqual(x[1, 0], -100)
        self.assertEqual(x[1, 8], 0)

        np.testing.assert_array_equal(y, [[1.0, 2.0], [4.0, 5.0]])
        np.testing.assert_array_equal(capture_ids, [1,2])


if __name__ == "__main__":
    unittest.main()