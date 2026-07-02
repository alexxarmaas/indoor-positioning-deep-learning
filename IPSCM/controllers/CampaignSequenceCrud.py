from sqlalchemy import func
from models.BeaconBleSignal import BeaconBleSignal
from models.Campaign import Campaign
from models.CampaignConfigurations import CampaignConfigurations
from models.CampaignSequence import CampaignSequence, CampaignSequenceAux
from models.Capture import Capture
from config import db

def createCampaignSequence():
    db.create_all()
    sequenceByName = {
        sequence.Name: sequence.Seq
        for sequence in CampaignSequenceAux.query.all()
    }

    campaignSequence = CampaignSequence()
    campaignSequence.Dongle_receptor = sequenceByName.get('Dongle_Receptor', 0)
    campaignSequence.Beacon_configuration = sequenceByName.get('Beacon_Configuration', 0)
    campaignSequence.Capture = sequenceByName.get('Capture', 0)
    campaignSequence.Beacon_BLE_Signal = sequenceByName.get('Beacon_BLE_Signal', 0)
    db.session.add(campaignSequence)

    db.session.commit()
    db.session.remove()

def deleteCampaignSequenceByCampaignId(sequenceId):
    existingSequence = CampaignSequence().query.filter_by(Id=sequenceId).first()

    if existingSequence:
        db.session.delete(existingSequence)
        db.session.commit()

def getCampaignSequence(id: int):
    sequence = CampaignSequence.query.filter_by(Id=id).first()
    if sequence is None:
        return None

    data = sequence.serialize()
    campaign = Campaign.query.filter_by(Id_campaign_sequence=id).first()
    if campaign is None:
        return data

    data['Capture'] = Capture.query.filter_by(Id_campaign=campaign.Id).count()
    data['Beacon_BLE_Signal'] = (
        db.session.query(func.count(BeaconBleSignal.Id))
        .join(Capture, BeaconBleSignal.Id_capture == Capture.Id)
        .filter(Capture.Id_campaign == campaign.Id)
        .scalar()
    )
    data['Beacon_configuration'] = CampaignConfigurations.query.filter_by(
        Id_campaign=campaign.Id
    ).count()
    return data