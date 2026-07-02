from flask import current_app, request
from flask_restful import Resource
from controllers.CampaignCrud import *
from controllers.BeaconBleSignalCrud import getBeaconBleSignalByCampaignId
from controllers.BeaconConfigurationCrud import getJsonBeaconConfigurationByCampaignId
from controllers.CampaignSequenceCrud import getCampaignSequence
from controllers.CaptureCrud import getCapturesByCampaignId
from controllers.DongleReceptorCrud import getDongleReceptor
from controllers.MethodPredictionCrud import getPredictionsByCampaignId
from controllers.TaskHistoryCrud import getTaskHistory
from services.DataProcessService import dataProcessing
from services.UploadService import uploadCampaign
from services.ImageService import generateZipImages
from services.ImportProgressService import completeImport, failImport, getImportProgress, releaseImport, startImport

def _getPaginationArguments():
    page = request.args.get('page', type=int)
    pageSize = request.args.get('pageSize', type=int)

    if page is None and pageSize is None:
        return None, None

    return page or 0, pageSize or 10

class ServerStatus(Resource):
    def get(self):
        return {"status": "Server is running"}

class GetCampaigns(Resource):
    def get(self):
        return getCampaigns()


class GetImportProgress(Resource):
    def get(self):
        return getImportProgress()
    
class GetTaskHistory(Resource):
    def get(self):
        return getTaskHistory()

class GetCampaignImagesById(Resource):
    def get(self, campaignId):
        return getCampaignImagesById(campaignId)
    
class GetRelatedCampaignById(Resource):
    def get(self, relatedCampaignId):
        return getRelatedCampaignById(relatedCampaignId)
    
class GetDongleReceptor(Resource):
    def get(self, dongleId):
        return getDongleReceptor(dongleId)
    
class GetCampaignSequence(Resource):
    def get(self, sequenceId):
        return getCampaignSequence(sequenceId)
    
class GetCapturesByCampaignId(Resource):
    def get(self, campaignId):
        page, pageSize = _getPaginationArguments()
        return getCapturesByCampaignId(campaignId, page, pageSize)
    
class GetBeaconBleSignalByCampaignId(Resource):
    def get(self, campaignId):
        page, pageSize = _getPaginationArguments()
        return getBeaconBleSignalByCampaignId(campaignId, page, pageSize)

class GetBeaconConfigurationByCampaignId(Resource):
    def get(self, campaignId):
        return getJsonBeaconConfigurationByCampaignId(campaignId)
    
class GetPredictionsByCampaignId(Resource):
    def get(self, campaignId):
        return getPredictionsByCampaignId(campaignId)   
    
class DataProcessing(Resource):
    def post(self):
        params = request.form.get('params')
        data =  json.loads(params)
        return dataProcessing(data)
    
class UploadCampaign(Resource):
    def post(self):
        name = request.form.get('name')
        date = request.form.get('date')
        imagesRef = None

        description = request.form.get('description')
        if description=='null':
            description = "No hay descripción"
        
        if 'images' in request.files:
            images = request.files.getlist('images')
            imagesRef = generateZipImages(images)
        
        aleatoryFile = request.files.get('aleatoryFile')
        referenceFiles = request.files.getlist('referenceFiles')
        confs = request.files.getlist('confs')

        if aleatoryFile is None or not referenceFiles:
            return {'message': 'Carga una base aleatoria y al menos una de referencia'}, 400

        if not startImport(1 + len(referenceFiles)):
            return {
                'message': 'Ya hay otra importacion en curso',
                'progress': getImportProgress()
            }, 409

        try:
            result = uploadCampaign(
                name, date, description, imagesRef,
                aleatoryFile, referenceFiles, confs
            )
            completeImport('Campana importada correctamente')
            return result, 201
        except Exception:
            current_app.logger.exception('Campaign import failed')
            failImport('La importacion ha fallado. Revisa los logs del servidor.')
            return {'message': 'La importacion ha fallado'}, 500
        finally:
            releaseImport()
    
class DeteteCampaignById(Resource):
    def delete(self, campaignId):
        return deleteCampaignById(campaignId)
