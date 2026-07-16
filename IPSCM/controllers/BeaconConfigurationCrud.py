import logging
from flask import jsonify
from controllers.BeaconBleSignalCrud import getMacsByCampaignId, normalizeMac
from controllers.CampaignConfigurationsCrud import createCampaignConfigurations, getCampaignBeaconConfigurationIdsByCampaignId
from models.BeaconConfiguration import BeaconConfiguration, BeaconConfigurationAux
from config import db
from sqlalchemy import func

logger = logging.getLogger(__name__)


def _normalizedMacColumn(column):
    return func.replace(func.lower(func.trim(column)), '-', ':')


def _sourceConfigurationsByMac():
    configurations = {}
    for source in BeaconConfigurationAux.query.order_by(BeaconConfigurationAux.Id.asc()).all():
        mac = normalizeMac(source.Mac)
        if mac and mac not in configurations:
            configurations[mac] = source
    return configurations


def _findExistingConfiguration(mac, source):
    candidates = BeaconConfiguration.query.filter(
        _normalizedMacColumn(BeaconConfiguration.Mac) == mac
    ).order_by(BeaconConfiguration.Id.asc()).all()

    if source is not None:
        for candidate in candidates:
            if (
                candidate.Power == source.Power
                and candidate.Update_frequency == source.Update_frequency
            ):
                return candidate
        return None

    return candidates[0] if candidates else None


def _createObservedConfiguration(mac, source):
    configuration = BeaconConfiguration()
    configuration.Mac = mac

    if source is not None:
        configuration.Power = source.Power
        configuration.Calibrated_power = source.Calibrated_power
        configuration.Update_frequency = source.Update_frequency
        configuration.Identificator = source.Identificator

    db.session.add(configuration)
    db.session.flush()
    return configuration


def createBeaconConfiguration(campaignId, beaconConfName):
    """Associate every MAC observed in the imported campaign."""
    db.create_all()

    sourcesByMac = _sourceConfigurationsByMac()
    observedMacs = getMacsByCampaignId(campaignId)
    missingMetadata = []

    for mac in observedMacs:
        source = sourcesByMac.get(mac)
        configuration = _findExistingConfiguration(mac, source)

        if configuration is None:
            configuration = _createObservedConfiguration(mac, source)

        createCampaignConfigurations(
            campaignId, configuration.Id, beaconConfName
        )

        if source is None:
            missingMetadata.append(mac)

    db.session.commit()
    db.session.remove()

    if missingMetadata:
        logger.warning(
            'Campaign %s imported %s observed beacons without configuration '
            'metadata: %s',
            campaignId, len(missingMetadata), ', '.join(missingMetadata)
        )

    return {
        'observedCount': len(observedMacs),
        'missingMetadata': missingMetadata
    }

def getJsonBeaconConfigurationByCampaignId(campaignId):
    ids = getCampaignBeaconConfigurationIdsByCampaignId(campaignId)
    beacon_configurations = BeaconConfiguration.query.filter(BeaconConfiguration.Id.in_(ids)).all()
    return jsonify([beacon_configuration.serialize() for beacon_configuration in beacon_configurations])

def getBeaconConfigurationByCampaignId(campaignId):
    ids = getCampaignBeaconConfigurationIdsByCampaignId(campaignId)
    beacon_configurations = BeaconConfiguration.query.filter(BeaconConfiguration.Id.in_(ids)).all()
    return [beacon_configuration.Mac for beacon_configuration in beacon_configurations]


def getIdByMacPowerFrequency(mac, power, frequency):
    return BeaconConfiguration.query.filter_by(Mac=mac, Power=power, Update_frequency=frequency).first().Id

def getLastInsertedBeaconConfigurationId():
    return BeaconConfiguration.query.order_by(BeaconConfiguration.Id.desc()).first().Id
