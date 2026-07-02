import json
from flask import jsonify
from datetime import datetime
from config import db
from controllers.BeaconBleSignalCrud import deleteBeaconBleSignalByCampaignId
from controllers.CampaignConfigurationsCrud import deleteConfigurationsByCampaignId
from controllers.MethodPredataCrud import deleteMethodPredataByCampaignId
from controllers.MethodPredictionCrud import deleteMethodPredictionByCampaignId
from controllers.TaskHistoryCrud import deleteTaskHistoryByCampaignId
from models.DongleReceptor import DongleReceptor, DongleReceptorAux
from models.Campaign import Campaign
from models.CampaignSequence import CampaignSequence
from controllers.CampaignSequenceCrud import deleteCampaignSequenceByCampaignId
from controllers.CaptureCrud import deleteCaptureByCampaignId
from sqlalchemy import desc
from services.ImageService import deleteZipImages, sendZipImages


def createCampaign(name, date, description, images, relatedCampaignId, campaignParams,alePointsJson, refPointsJson):
    db.create_all()
    auxDongleName = DongleReceptorAux().query.first().Name
    dongleId = DongleReceptor().query.filter_by(Name=auxDongleName).first().Id
    campaignSequenceId = CampaignSequence().query.order_by(desc(CampaignSequence.Id)).first().Id
    formattedDate = datetime.strptime(date, '%Y-%m-%d').date()
    
    campaign = Campaign()
    campaign.Id_related_campaign = relatedCampaignId
    campaign.Id_dongle = dongleId
    campaign.Id_campaign_sequence = campaignSequenceId
    campaign.Name = name
    campaign.Description = description
    campaign.Date = formattedDate
    campaign.Images_ref = images
    campaign.Duration = campaignParams[0]
    campaign.Rotations = campaignParams[1]
    campaign.Ale_points = alePointsJson
    campaign.Ref_points = refPointsJson

    db.session.add(campaign)

    db.session.commit()
    campaignId = campaign.Id
    db.session.remove()
    return campaignId


def getCampaigns():
    campaigns = Campaign.query.filter(Campaign.Id_related_campaign != None).all()
    return jsonify([campaign.serialize() for campaign in campaigns])

def getRelatedCampaignById(relatedCampaignId):
    campaign = Campaign.query.filter_by(Id=relatedCampaignId).first()
    return jsonify(campaign.serialize())

def getCampaignImagesById(campaignId):
    campaign = Campaign.query.filter_by(Id=campaignId).first()

    if campaign.Images_ref is None:
        return {'message': 'Campaign has no image'}, 404

    response = sendZipImages(campaign.Images_ref)
    return response

def deleteCampaignById(campaignId):
    campaign = Campaign().query.filter_by(Id=campaignId).first()
    relatedCampaign = Campaign().query.filter_by(Id=campaign.Id_related_campaign).first()

    if campaign:
        deleteCampaignSequenceByCampaignId(campaignId)
        deleteCampaignSequenceByCampaignId(relatedCampaign.Id)

        deleteBeaconBleSignalByCampaignId(campaignId)
        deleteBeaconBleSignalByCampaignId(relatedCampaign.Id)

        deleteCaptureByCampaignId(campaignId)
        deleteCaptureByCampaignId(relatedCampaign.Id)

        deleteConfigurationsByCampaignId(campaignId)
        deleteConfigurationsByCampaignId(relatedCampaign.Id)

        deleteMethodPredataByCampaignId(campaignId)
        deleteMethodPredataByCampaignId(relatedCampaign.Id)

        deleteMethodPredictionByCampaignId(campaignId)

        deleteTaskHistoryByCampaignId(campaignId)

        if campaign.Images_ref is not None:
            deleteZipImages(campaign.Images_ref)

        db.session.delete(campaign)
        db.session.delete(relatedCampaign)

        db.session.commit()
        db.session.execute('VACUUM')

        return {'message': 'Campaign successfully deleted'}, 200
    else:
        return {'message', 'Campaign not found'}, 404
    
def getLastInsertedCampaignId():
    return Campaign.query.order_by(Campaign.Id.desc()).first().Id


def getRelatedCampaignIdByCampaignId(campaignId):
    campaign = Campaign.query.filter_by(Id=campaignId).first()

    if campaign and campaign.Id_related_campaign is not None:
        return campaign.Id_related_campaign

    raise ValueError(f'Campaign {campaignId} has no related aleatory campaign')


def getPointsByCampaignId(campaignId):
    campaign = Campaign.query.filter_by(Id=campaignId).first()

    if campaign:
        return {
            'Ale_points': json.loads(campaign.Ale_points),
            'Ref_points': json.loads(campaign.Ref_points)
        }
