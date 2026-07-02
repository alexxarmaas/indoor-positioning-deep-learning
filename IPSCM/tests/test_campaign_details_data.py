import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from controllers.CampaignSequenceCrud import createCampaignSequence, getCampaignSequence


class CampaignDetailsDataTest(unittest.TestCase):
    def test_campaign_sequence_uses_table_names_instead_of_row_positions(self):
        rows = [
            SimpleNamespace(Name='Dongle_Receptor', Seq=1),
            SimpleNamespace(Name='Beacon_Configuration', Seq=30),
            SimpleNamespace(Name='Campaign_Configurations', Seq=17),
            SimpleNamespace(Name='Capture', Seq=60),
            SimpleNamespace(Name='Beacon_BLE_Signal', Seq=46693),
        ]
        sequence = SimpleNamespace()
        fake_session = SimpleNamespace(
            add=MagicMock(), commit=MagicMock(), remove=MagicMock()
        )
        fake_db = SimpleNamespace(create_all=MagicMock(), session=fake_session)
        fake_aux = SimpleNamespace(
            query=SimpleNamespace(all=MagicMock(return_value=rows))
        )
        fake_sequence_model = MagicMock(return_value=sequence)

        with patch('controllers.CampaignSequenceCrud.db', fake_db), \
             patch('controllers.CampaignSequenceCrud.CampaignSequenceAux', fake_aux), \
             patch('controllers.CampaignSequenceCrud.CampaignSequence', fake_sequence_model):
            createCampaignSequence()

        self.assertEqual(sequence.Dongle_receptor, 1)
        self.assertEqual(sequence.Beacon_configuration, 30)
        self.assertEqual(sequence.Capture, 60)
        self.assertEqual(sequence.Beacon_BLE_Signal, 46693)

    def test_campaign_sequence_response_uses_live_aggregate_counts(self):
        stored_sequence = SimpleNamespace(serialize=lambda: {
            'Id': 2,
            'Dongle_receptor': 1,
            'Beacon_configuration': 30,
            'Capture': 70,
            'Beacon_BLE_Signal': 240,
        })
        sequence_query = MagicMock()
        sequence_query.filter_by.return_value.first.return_value = stored_sequence
        fake_sequence_model = SimpleNamespace(query=sequence_query)

        campaign_query = MagicMock()
        campaign_query.filter_by.return_value.first.return_value = SimpleNamespace(Id=2)
        fake_campaign_model = SimpleNamespace(query=campaign_query)

        capture_query = MagicMock()
        capture_query.filter_by.return_value.count.return_value = 1944
        fake_capture_model = SimpleNamespace(
            query=capture_query, Id='capture-id', Id_campaign='campaign-id'
        )

        signal_count_query = MagicMock()
        signal_count_query.join.return_value = signal_count_query
        signal_count_query.filter.return_value = signal_count_query
        signal_count_query.scalar.return_value = 1369782
        fake_db = SimpleNamespace(
            session=SimpleNamespace(query=MagicMock(return_value=signal_count_query))
        )
        fake_signal_model = SimpleNamespace(
            Id='signal-id', Id_capture='signal-capture-id'
        )

        config_query = MagicMock()
        config_query.filter_by.return_value.count.return_value = 30
        fake_config_model = SimpleNamespace(query=config_query)

        with patch('controllers.CampaignSequenceCrud.CampaignSequence', fake_sequence_model), \
             patch('controllers.CampaignSequenceCrud.Campaign', fake_campaign_model), \
             patch('controllers.CampaignSequenceCrud.Capture', fake_capture_model), \
             patch('controllers.CampaignSequenceCrud.BeaconBleSignal', fake_signal_model), \
             patch('controllers.CampaignSequenceCrud.CampaignConfigurations', fake_config_model), \
             patch('controllers.CampaignSequenceCrud.db', fake_db):
            result = getCampaignSequence(2)

        self.assertEqual(result['Capture'], 1944)
        self.assertEqual(result['Beacon_BLE_Signal'], 1369782)
        self.assertEqual(result['Beacon_configuration'], 30)


if __name__ == '__main__':
    unittest.main()