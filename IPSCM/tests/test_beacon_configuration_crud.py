import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from controllers.BeaconConfigurationCrud import createBeaconConfiguration


class BeaconConfigurationCrudTest(unittest.TestCase):
    def test_import_associates_every_observed_mac(self):
        source = SimpleNamespace(Mac='aa:aa')
        configured = SimpleNamespace(Id=1)
        placeholder = SimpleNamespace(Id=2)

        with (
            patch('controllers.BeaconConfigurationCrud.db') as db,
            patch(
                'controllers.BeaconConfigurationCrud._sourceConfigurationsByMac',
                return_value={'aa:aa': source}
            ),
            patch(
                'controllers.BeaconConfigurationCrud.getMacsByCampaignId',
                return_value=['aa:aa', 'bb:bb']
            ),
            patch(
                'controllers.BeaconConfigurationCrud._findExistingConfiguration',
                side_effect=[configured, None]
            ),
            patch(
                'controllers.BeaconConfigurationCrud._createObservedConfiguration',
                return_value=placeholder
            ) as create_observed,
            patch(
                'controllers.BeaconConfigurationCrud.createCampaignConfigurations'
            ) as associate,
            patch('controllers.BeaconConfigurationCrud.logger')
        ):
            report = createBeaconConfiguration(7, 'Configuracion 1')

        associate.assert_has_calls([
            call(7, 1, 'Configuracion 1'),
            call(7, 2, 'Configuracion 1'),
        ])
        create_observed.assert_called_once_with('bb:bb', None)
        db.session.commit.assert_called_once_with()
        self.assertEqual(report, {
            'observedCount': 2,
            'missingMetadata': ['bb:bb']
        })


if __name__ == '__main__':
    unittest.main()
