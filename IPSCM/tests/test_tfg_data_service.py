import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from services.TFGDataService import load_tfg_campaign_dataset
from services.TFGFeaturesService import load_beacon_mapping

class TestTFGDataService(unittest.TestCase):

    @patch("services.TFGDataService.getCaptureIdsByCampaignId")
    @patch("services.TFGDataService.getCapturesById")
    @patch("services.TFGDataService.getBeaconBleSignalsByCaptureIds")

    def test_load_campaign_dataset(self, mock_signals, mock_captures, mock_capture_ids):

        mapping = load_beacon_mapping()

        mock_capture_ids.return_value = [1,2]
        mock_captures.return_value = [
            SimpleNamespace(
                Id=1,
                Position_x = 1.0,
                Position_y = 2.0
            ),
            SimpleNamespace(
                Id=2,
                Position_x = 4.0,
                Position_y = 5.0
            )
        ]

        mock_signals.return_value = [
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

        X, y , ids = load_tfg_campaign_dataset(10)

        self.assertEqual(X.shape, (2, 16))
        self.assertEqual(y.shape, (2, 2))

        self.assertEqual(X[0, 0], -65)
        self.assertEqual(X[1, 1], -80)

        np.testing.assert_array_equal(y, [[1.0, 2.0], [4.0, 5.0]])
        np.testing.assert_array_equal(ids, [1, 2])

        mock_capture_ids.assert_called_once_with(10)
        mock_captures.assert_called_once_with([1, 2])
        mock_signals.assert_called_once_with([1, 2])

    @patch("services.TFGDataService.getCaptureIdsByCampaignId")
    @patch("services.TFGDataService.getCapturesById")
    @patch("services.TFGDataService.getBeaconBleSignalsByCaptureIds")
    def test_empty_capture_is_excluded(self, mock_signals, mock_captures, mock_capture_ids):
        mapping = load_beacon_mapping()

        mock_capture_ids.return_value = [1, 2, 3]

        mock_captures.return_value = [
            SimpleNamespace(Id=1, Position_x=1.0, Position_y=2.0),
            SimpleNamespace(Id=2, Position_x=3.0, Position_y=4.0),
            SimpleNamespace(Id=3, Position_x=5.0, Position_y=6.0),
        ]

        mock_signals.return_value = [
            SimpleNamespace(Id_capture=1, Mac=mapping["b01"], RSSI=-60),
            SimpleNamespace(Id_capture=3, Mac=mapping["b02"], RSSI=-80)
        ]

        X, y, ids = load_tfg_campaign_dataset(10)

        self.assertEqual(X.shape, (2, 16))
        self.assertEqual(y.shape, (2, 2))

        np.testing.assert_array_equal(ids, [1,3])
        np.testing.assert_array_equal(y, [[1.0, 2.0], [5.0, 6.0]])

if __name__ == "__main__":
    unittest.main()