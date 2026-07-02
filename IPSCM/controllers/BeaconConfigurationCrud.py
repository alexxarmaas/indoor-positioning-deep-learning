from flask import jsonify
from controllers.BeaconBleSignalCrud import getMacsByCampaignId
from controllers.CampaignConfigurationsCrud import createCampaignConfigurations, getCampaignBeaconConfigurationIdsByCampaignId
from models.BeaconConfiguration import BeaconConfiguration, BeaconConfigurationAux
from config import db
from sqlalchemy import desc, func

def createBeaconConfiguration(campaignId, beaconConfName):
    db.create_all()

    unique_macs = db.session.query(BeaconConfigurationAux.Mac).distinct().all()
    uniqueMacsList = [x[0] for x in unique_macs]
    actualBeaconConfigurations = db.session.query(BeaconConfiguration).all()
    newBeaconConfigurations = BeaconConfigurationAux.query.filter(BeaconConfigurationAux.Mac.in_(uniqueMacsList)).group_by(BeaconConfigurationAux.Mac).all()
    usedBeacons = getMacsByCampaignId(campaignId)


    for i in newBeaconConfigurations:
        is_unique = True
        for j in actualBeaconConfigurations:
            if  i.Mac == j.Mac and i.Power == j.Power and i.Update_frequency == j.Update_frequency:
                if i.Mac in usedBeacons:
                    beaconConfId = getIdByMacPowerFrequency(i.Mac, i.Power, i.Update_frequency)
                    createCampaignConfigurations(campaignId, beaconConfId, beaconConfName)
                is_unique = False
                break
        if is_unique:
                beaconConfiguration = BeaconConfiguration()
                beaconConfiguration.Mac = i.Mac
                beaconConfiguration.Power = i.Power
                beaconConfiguration.Calibrated_power = i.Calibrated_power
                beaconConfiguration.Update_frequency = i.Update_frequency
                beaconConfiguration.Identificator = i.Identificator
                db.session.add(beaconConfiguration)
                db.session.flush()

                if i.Mac in usedBeacons:
                    beaconConfId = beaconConfiguration.Id
                    createCampaignConfigurations(campaignId, beaconConfId, beaconConfName)

    db.session.commit()
    db.session.remove()


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
