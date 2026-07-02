import json
import unittest
from types import SimpleNamespace
from unittest.mock import ANY, call, patch

from controllers.CaptureCrud import buildCoordinateMapping
from services.DataProcessService import (
    MISSING_RSSI,
    getAlePointsMatrix,
    getRefPointsMatrix,
)
from services.ImportProgressService import completeImport, getImportProgress, releaseImport, startImport, updateImportProgress
from services.UploadService import uploadCampaign


class CampaignAggregationTest(unittest.TestCase):
    def test_coordinates_from_different_sessions_are_normalized_by_point_order(self):
        captures = [
            SimpleNamespace(Position_x=0.71, Position_y=0.32, Position_z=0.0),
            SimpleNamespace(Position_x=0.71, Position_y=0.32, Position_z=0.0),
            SimpleNamespace(Position_x=1.69, Position_y=0.34, Position_z=0.0),
        ]
        configuration = {
            'POINT_0': {'X': '0.674', 'Y': '0.316', 'Z': '0'},
            'POINT_1': {'X': '1.684', 'Y': '0.327', 'Z': '0'},
        }

        mapping = buildCoordinateMapping(captures, configuration)

        self.assertEqual(mapping[(0.71, 0.32, 0.0)], (0.674, 0.316, 0.0))
        self.assertEqual(mapping[(1.69, 0.34, 0.0)], (1.684, 0.327, 0.0))

    def test_database_with_more_points_than_configuration_is_rejected(self):
        captures = [
            SimpleNamespace(Position_x=0.0, Position_y=0.0, Position_z=0.0),
            SimpleNamespace(Position_x=1.0, Position_y=0.0, Position_z=0.0),
        ]
        configuration = {
            'POINT_0': {'X': '0', 'Y': '0', 'Z': '0'},
        }

        with self.assertRaises(ValueError):
            buildCoordinateMapping(captures, configuration)

    @patch('services.DataProcessService.getFilteredPredataSamples')
    def test_reference_matrix_averages_rotation_maxima_and_fills_missing_rssi(
        self, getSamples
    ):
        getSamples.side_effect = [
            [SimpleNamespace(RSSI=-80), SimpleNamespace(RSSI=-60)],
            [SimpleNamespace(RSSI=-90), SimpleNamespace(RSSI=-50)],
            [],
            [],
        ]
        points = [{'x': 0.0, 'y': 0.0, 'z': 0.0}]
        configuration = {
            'POINT_0': {
                'X': '0', 'Y': '0', 'Z': '0', 'Rotations': ['0', '90']
            }
        }

        matrix = getRefPointsMatrix(
            points, configuration, ['visible', 'missing'], 1,
            'iBeacon', 37, 2
        )

        self.assertEqual(matrix[0, 0], -55.0)
        self.assertEqual(matrix[0, 1], MISSING_RSSI)
        self.assertEqual(matrix[0, -3:].tolist(), [0.0, 0.0, 0.0])

    @patch('services.DataProcessService.getFilteredPredataSamples')
    def test_aleatory_matrix_uses_the_maximum_limited_sample(self, getSamples):
        getSamples.side_effect = [
            [SimpleNamespace(RSSI=-75), SimpleNamespace(RSSI=-42)],
            [],
        ]
        points = [{'x': 1.0, 'y': 2.0, 'z': 0.0}]
        configuration = {
            'POINT_0': {'X': '1', 'Y': '2', 'Z': '0', 'Rotation': '90'}
        }

        matrix = getAlePointsMatrix(
            points, configuration, ['visible', 'missing'], 2,
            'iBeacon', 37, 2
        )

        self.assertEqual(matrix[0, 0], -42.0)
        self.assertEqual(matrix[0, 1], MISSING_RSSI)

    def test_one_visible_campaign_receives_all_reference_databases(self):
        point_configuration = json.dumps({
            'POINT_0': {'X': '0', 'Y': '0', 'Z': '0'}
        })
        databases = ['aleatory', 'reference-1', 'reference-2', 'reference-3']
        configurations = ['general', 'aleatory-points', 'reference-points']

        with patch('services.UploadService.readBLEConf', return_value=['15', '0,90']), \
             patch('services.UploadService.readAlePointsConf', return_value=point_configuration), \
             patch('services.UploadService.readRefPointsConf', return_value=point_configuration), \
             patch('services.UploadService._createCampaignFromDatabase', side_effect=[101, 202]) as create, \
             patch('services.UploadService._appendDatabaseToCampaign') as append, \
             patch('services.UploadService.generatePredata') as generate:
            result = uploadCampaign(
                'Aggregated campaign', '2026-06-29', '', None,
                databases, configurations
            )

        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0].args[0], 'aleatory')
        self.assertEqual(create.call_args_list[1].args[0], 'reference-1')
        self.assertEqual(create.call_args_list[1].args[5], 101)
        self.assertEqual(
            append.call_args_list,
            [
                call('reference-2', 202, json.loads(point_configuration)),
                call('reference-3', 202, json.loads(point_configuration)),
            ]
        )
        self.assertEqual(generate.call_args_list, [call(101, ANY), call(202, ANY)])
        self.assertEqual(result['campaignId'], 202)
        self.assertEqual(result['referenceDatabases'], 3)

    def test_at_least_two_databases_are_required(self):
        with self.assertRaises(ValueError):
            uploadCampaign('Campaign', '2026-06-29', '', None, ['only-one'], [])

    def test_second_simultaneous_import_is_rejected_and_progress_is_visible(self):
        releaseImport()
        self.assertTrue(startImport(10))

        try:
            self.assertFalse(startImport(10))
            updateImportProgress(
                stage='reference-predata', message='Generating predata',
                percentage=73, current=730, total=1000
            )
            progress = getImportProgress()
            self.assertEqual(progress['state'], 'running')
            self.assertEqual(progress['percentage'], 73)
            self.assertEqual(progress['current'], 730)
            completeImport()
            self.assertEqual(getImportProgress()['state'], 'completed')
        finally:
            releaseImport()


if __name__ == '__main__':
    unittest.main()
