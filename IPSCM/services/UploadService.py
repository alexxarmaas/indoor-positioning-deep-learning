import os
import json
from controllers.BeaconConfigurationCrud import createBeaconConfiguration
from controllers.CampaignConfigurationsCrud import getLastInsertedBeaconConfName
from controllers.DongleReceptorCrud import createDongleReceptor
from controllers.BeaconBleSignalCrud import createBeaconBleSignal
from controllers.CaptureCrud import createCapture
from models.Capture import CaptureAux
from controllers.CampaignCrud import createCampaign
from controllers.CampaignSequenceCrud import createCampaignSequence
from services.ConfigReaderService import readBLEConf, readAlePointsConf, readRefPointsConf
from services.DataProcessService import generatePredata
from services.ImportProgressService import updateImportProgress

AUXILIARY_DATABASE_PATH = os.path.join('db', 'auxDB.sqlite3')


def uploadCampaign(name, date, description, imagesRef, aleatoryFile,
                   referenceFiles=None, confs=None):
    aleatoryFile, referenceFiles, confs = _normalizeUploadArguments(
        aleatoryFile, referenceFiles, confs
    )
    if aleatoryFile is None or not referenceFiles:
        raise ValueError('An aleatory and at least one reference database are required')
    if confs and len(confs) != 3:
        raise ValueError('Exactly three configuration files are required when provided')

    if confs:
        campaignParams = readBLEConf(confs[0])
        alePointsJson = readAlePointsConf(confs[1])
        refPointsJson = readRefPointsConf(confs[2])
        alePointsConfiguration = json.loads(alePointsJson)
        refPointsConfiguration = json.loads(refPointsJson)
    else:
        campaignParams = [None, None]
        alePointsJson = '{}'
        refPointsJson = '{}'
        alePointsConfiguration = None
        refPointsConfiguration = None

    updateImportProgress(
        stage='databases', message='Importando la base de puntos aleatorios...',
        percentage=2, current=0, total=1 + len(referenceFiles)
    )
    aleatoryCampaignId = _createCampaignFromDatabase(
        aleatoryFile, name, date, description, None, None, campaignParams,
        alePointsJson, refPointsJson, alePointsConfiguration
    )
    updateImportProgress(percentage=10, current=1)
    generatePredata(
        aleatoryCampaignId,
        _buildProgressCallback(
            10, 15, 'aleatory-predata',
            'Generando predatos de puntos aleatorios...'
        )
    )

    referenceCampaignId = None

    for index, referenceFile in enumerate(referenceFiles):
        updateImportProgress(
            stage='databases',
            message=f'Importando referencia {index + 1} de {len(referenceFiles)}...',
            percentage=round(15 + (45 * index / len(referenceFiles))),
            current=index + 1,
            total=1 + len(referenceFiles)
        )
        if index == 0:
            referenceCampaignId = _createCampaignFromDatabase(
                referenceFile, name, date, description, imagesRef,
                aleatoryCampaignId, campaignParams, alePointsJson,
                refPointsJson, refPointsConfiguration
            )
        else:
            _appendDatabaseToCampaign(
                referenceFile, referenceCampaignId, refPointsConfiguration
            )

        updateImportProgress(
            percentage=round(15 + (45 * (index + 1) / len(referenceFiles))),
            current=index + 2
        )

    generatePredata(
        referenceCampaignId,
        _buildProgressCallback(
            60, 99, 'reference-predata',
            'Generando predatos de todas las referencias...'
        )
    )

    return {
        'message': 'Campaign imported successfully',
        'campaignId': referenceCampaignId,
        'referenceDatabases': len(referenceFiles)
    }


def _normalizeUploadArguments(aleatoryFile, referenceFiles, confs):
    """Accept the explicit contract and the legacy combined file list."""
    if confs is None and isinstance(aleatoryFile, (list, tuple)):
        legacyFiles = list(aleatoryFile)
        confs = referenceFiles
        aleatoryFile = legacyFiles[0] if legacyFiles else None
        referenceFiles = legacyFiles[1:]

    return aleatoryFile, list(referenceFiles or []), list(confs or [])

def _createCampaignFromDatabase(dataFile, name, date, description, images,
                                relatedCampaignId, campaignParams,
                                alePointsJson, refPointsJson,
                                pointsConfiguration):
    _saveAuxiliaryDatabase(dataFile)

    try:
        if pointsConfiguration is None:
            pointsConfiguration = _buildIdentityPointsConfiguration()
            pointsJson = json.dumps(pointsConfiguration)
            if relatedCampaignId is None:
                alePointsJson = pointsJson
            else:
                refPointsJson = pointsJson

        createCampaignSequence()
        createDongleReceptor()
        campaignId = createCampaign(
            name, date, description, images, relatedCampaignId,
            campaignParams, alePointsJson, refPointsJson
        )
        _importAuxiliaryData(campaignId, pointsConfiguration)
        return campaignId
    finally:
        _removeAuxiliaryDatabase()


def _appendDatabaseToCampaign(dataFile, campaignId, pointsConfiguration):
    _saveAuxiliaryDatabase(dataFile)

    try:
        _importAuxiliaryData(campaignId, pointsConfiguration)
    finally:
        _removeAuxiliaryDatabase()


def _importAuxiliaryData(campaignId, pointsConfiguration):
    if pointsConfiguration is None:
        pointsConfiguration = _buildIdentityPointsConfiguration()

    lastBeaconConfName = getLastInsertedBeaconConfName()
    captureIdMapping = createCapture(campaignId, pointsConfiguration)
    createBeaconBleSignal(captureIdMapping)
    beaconConfName = getNextBeaconConfName(campaignId, lastBeaconConfName)
    createBeaconConfiguration(campaignId, beaconConfName)


def _buildIdentityPointsConfiguration():
    captures = CaptureAux.query.order_by(CaptureAux.Id.asc()).all()
    coordinates = []
    seenCoordinates = set()

    for capture in captures:
        coordinate = (capture.Position_x, capture.Position_y, capture.Position_z)
        if coordinate not in seenCoordinates:
            seenCoordinates.add(coordinate)
            coordinates.append(coordinate)

    return {
        f'POINT_{index}': {'X': str(x), 'Y': str(y), 'Z': str(z)}
        for index, (x, y, z) in enumerate(coordinates)
    }

def _saveAuxiliaryDatabase(dataFile):
    os.makedirs(os.path.dirname(AUXILIARY_DATABASE_PATH), exist_ok=True)
    dataFile.save(AUXILIARY_DATABASE_PATH)


def _removeAuxiliaryDatabase():
    if os.path.exists(AUXILIARY_DATABASE_PATH):
        os.remove(AUXILIARY_DATABASE_PATH)


def _buildProgressCallback(startPercentage, endPercentage, stage, message):
    def report(processed, total):
        fraction = processed / total if total else 1
        percentage = round(
            startPercentage + ((endPercentage - startPercentage) * fraction)
        )
        updateImportProgress(
            stage=stage,
            message=message,
            percentage=percentage,
            current=processed,
            total=total
        )

    return report


def getNextBeaconConfName(lastCampaignId, lastBeaconConfName):
    if lastBeaconConfName is not None:
        beaconConf = lastBeaconConfName.split('_')[:-1]
        try:
            index = int(lastBeaconConfName.split('_')[-1])
            index += 1
            if index >= 10:
                beaconConf.append(str(index))
            else:
                beaconConf.append('0' + str(index))
        except:
            index = lastCampaignId
            beaconConf.append('0' + str(index))
        
        return '_'.join(beaconConf)
    else:
        return "___Configuración___00"
