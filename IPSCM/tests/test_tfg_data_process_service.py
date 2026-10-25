import json
import unittest
from unittest.mock import patch

import numpy as np

from services.DataProcessService import dataProcessing


def matrix(points):
    features = np.zeros((len(points), 16), dtype=float)
    features[:, :8] = -65
    features[:, 8:16] = 1
    return np.column_stack((features, np.asarray(points, dtype=float)))


class TFGDataProcessServiceTest(unittest.TestCase):
    def test_process_uses_disjoint_campaigns_and_fuses_physical_points(self):
        train = matrix([(float(i), 0.0, 1.2) for i in range(12)])
        test = matrix([
            (20.0, 0.0, 1.2),
            (20.0, 0.0, 1.2),
            (22.0, 0.0, 1.2),
        ])
        with (
            patch("services.DataProcessService.getRelatedCampaignIdByCampaignId", return_value=2),
            patch("services.DataProcessService.build_tfg_campaign_matrix") as build,
            patch("services.DataProcessService.load_beacon_mapping", return_value={
                f"b{i:02d}": f"mac{i}" for i in range(1, 9)
            }),
            patch("services.DataProcessService.checkExistingMethodPrediction", return_value=False),
            patch("services.DataProcessService.createTaskHistory"),
            patch("services.DataProcessService.updateTaskHistory"),
            patch("services.DataProcessService.createMethodPrediction") as save,
            patch("services.DataProcessService.applyMethod") as apply,
        ):
            build.side_effect = [(train, list(range(100, 112))), (test, [200, 201, 202])]
            apply.return_value = {
                "[MLP_TFG_ROBUST]": [
                    [1.0, 2.0, 1.2],
                    [3.0, 4.0, 1.2],
                    [4.0, 5.0, 1.2],
                ]
            }
            result = dataProcessing({
                "campaign": 1,
                "campaignName": "TFG testing",
                "method": "MLP_TFG_ROBUST",
            })
            self.assertEqual(result["completedTasks"], 1)
            self.assertEqual(result["failedTasks"], 0)
            self.assertEqual(result["referenceCaptures"], 12)
            self.assertEqual(result["evaluationCaptures"], 3)
            self.assertEqual(result["evaluationPhysicalPoints"], 2)
            self.assertEqual(result["overlappingCoordinates"], 0)

            self.assertEqual([call.args[0] for call in build.call_args_list], [1, 2])
            self.assertEqual(apply.call_args.args[2], "MLP_TFG_ROBUST")
            saved = json.loads(save.call_args.args[2])
            predictions = next(iter(saved.values()))
            np.testing.assert_allclose(predictions, [
                [2.0, 3.0, 1.2],
                [4.0, 5.0, 1.2],
            ])

    def test_reuses_original_wknn_method_without_resampling(self):
        train = matrix([(float(i), 0.0, 1.2) for i in range(12)])
        test = matrix([(20.0, 0.0, 1.2)])
        with (
            patch("services.DataProcessService.getRelatedCampaignIdByCampaignId", return_value=2),
            patch("services.DataProcessService.build_tfg_campaign_matrix") as build,
            patch("services.DataProcessService.load_beacon_mapping", return_value={
                f"b{i:02d}": f"mac{i}" for i in range(1, 9)
            }),
            patch("services.DataProcessService.checkExistingMethodPrediction", return_value=False),
            patch("services.DataProcessService.createTaskHistory"),
            patch("services.DataProcessService.updateTaskHistory"),
            patch("services.DataProcessService.createMethodPrediction"),
            patch("services.DataProcessService.applyMethod") as apply,
        ):
            build.side_effect = [(train, list(range(100, 112))), (test, [200])]
            apply.return_value = {"[WKNN_TFG][k=2]": [[20.0, 0.0, 1.2]]}
            result = dataProcessing({
                "campaign": 1, "campaignName": "TFG testing",
                "method": "WKNN_TFG",
                "metrics": ["euclidean"], "algorithms": ["brute"],
                "kRangeStart": 2, "kRangeEnd": 3,
            })
            self.assertEqual(result["completedTasks"], 1)
            self.assertEqual(apply.call_args.args[2], "WKNN_TFG")
            self.assertEqual(apply.call_args.args[5], [2, 3])


if __name__ == "__main__":
    unittest.main()
