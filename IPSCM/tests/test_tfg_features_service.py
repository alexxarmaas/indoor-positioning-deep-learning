import unittest
from services.TFGFeaturesService import (
    load_beacon_mapping,
    build_tfg_fingerprint, 
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

if __name__ == "__main__":
    unittest.main()