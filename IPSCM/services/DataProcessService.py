import json
import numpy as np
from datetime import datetime
from controllers.BeaconBleSignalCrud import (
    countBeaconBleSignalsByCaptureIds, getBeaconBleSignalById,
    getObservedMacsByCampaignAndProtocol
)
from controllers.CampaignCrud import getPointsByCampaignId, getRelatedCampaignIdByCampaignId
from controllers.CaptureCrud import getCaptureIdsByCampaignId, getCapturesById
from controllers.MethodPredataCrud import createMethodPredataBatch, getFilteredPredataSamples, getUniqueCoordinatesByCampaignId
from controllers.MethodPredictionCrud import checkExistingMethodPrediction, createMethodPrediction
from services.HybridPositioningService import (
    applyHybridDeepLearningMethod,
    getHybridDeepLearningConfiguration,
)
from services.MethodsService import (
    DL_AGGREGATE_FINGERPRINT_REPETITIONS,
    MISSING_RSSI,
    applyMethod,
    calculateError,
    getDeepLearningConfiguration,
    getPyTorchDeepLearningConfiguration,
    validateFiniteResults,
)
from controllers.TaskHistoryCrud import createTaskHistory, updateTaskHistory

def generatePredata(campaignId, progressCallback=None):
    captureIds = getCaptureIdsByCampaignId(campaignId)
    totalSignals = countBeaconBleSignalsByCaptureIds(captureIds)
    processedSignals = 0
    filteredCapture = [
        (
            capture.Id, capture.Date, capture.Dongle_rotation,
            capture.Position_x, capture.Position_y, capture.Position_z
        )
        for capture in getCapturesById(captureIds)
    ]

    batch_data = []

    for captureId, date, rotation, positionX, positionY, positionZ in filteredCapture:
        filteredSignals = [
            (signal.Channel, signal.Mac, signal.Protocol, signal.RSSI)
            for signal in getBeaconBleSignalById(captureId)
        ]
        for channel, mac, protocol, rssi in filteredSignals:
            batch_data.append((campaignId, date, channel, mac, rotation, protocol,
                             rssi, positionX, positionY, positionZ))
            if len(batch_data) == 10000:
                batchSize = len(batch_data)
                createMethodPredataBatch(batch_data)
                batch_data.clear()
                processedSignals += batchSize
                if progressCallback:
                    progressCallback(processedSignals, totalSignals)

    if batch_data:
        batchSize = len(batch_data)
        createMethodPredataBatch(batch_data)
        processedSignals += batchSize

    if progressCallback:
        progressCallback(processedSignals, totalSignals)

def dataProcessing(data):
    campaignId = data['campaign']
    campaignName = data['campaignName']
    method = data['method']
    protocols = data['protocols']
    channels = data['channels']
    rssiSamples = data['sample']
    metrics = data['metrics']
    algorithms = data['algorithms']
    kernels = data['kernels']
    cs = data['cValues']
    gammas = data['gValues']
    nus = data['nuValues']
    Is = data['iValues']
    ksRange = []
    if method == 'DeepLearningHybrid':
        return processDeepLearningHybrid(data)

    ksRange.append(data['kRangeStart'])

    configPoints = getPointsByCampaignId(campaignId)
    aleCampaignId = getRelatedCampaignIdByCampaignId(campaignId)

    ref_points_conf = configPoints['Ref_points']
    ale_points_conf = configPoints['Ale_points']

    ref_points = getUniqueCoordinatesByCampaignId(campaignId, len(ref_points_conf))
    ale_points = getUniqueCoordinatesByCampaignId(aleCampaignId, len(ale_points_conf))

    if(len(ref_points)) < data['kRangeEnd']:
        ksRange.append(len(ref_points))
    else:
        ksRange.append(data['kRangeEnd'])
    
    ksRangeJson = json.dumps(ksRange)
    completedTasks = 0
    skippedTasks = 0
    failedTasks = 0

    for protocol in protocols:
        beaconMacs = processBeaconMacs(campaignId, aleCampaignId, protocol)
        if not beaconMacs:
            raise ValueError(f'No common beacons found for protocol {protocol}')
        for channel in channels:
            for metric in metrics:
                for algorithm in algorithms:
                    for kernel in kernels:

                        taskId = taskToJson(method, protocol, channel, rssiSamples, algorithm, metric, ksRangeJson, kernel, cs, gammas, nus, Is, beaconMacs)
                        
                        if not checkExistingMethodPrediction(campaignId, taskId):

                            try:
                                datetime_start = datetime.now()
                                createTaskHistory(campaignId, campaignName, "Obteniendo matriz de puntos de referencia...", taskId, datetime_start)
                                
                                if method in (
                                    'DeepLearning',
                                    'DeepLearningPyTorch',
                                ):
                                    aleatoryRotations = {
                                        float(point['Rotation'])
                                        for point in ale_points_conf.values()
                                    }
                                    refRSSIMatrix = getDeepLearningRefPointsMatrix(
                                        ref_points, ref_points_conf, beaconMacs,
                                        campaignId, protocol, channel, rssiSamples,
                                        aleatoryRotations
                                    )
                                else:
                                    refRSSIMatrix = getRefPointsMatrix(
                                        ref_points, ref_points_conf, beaconMacs,
                                        campaignId, protocol, channel, rssiSamples
                                    )

                                updateTaskHistory(campaignId, taskId, "Obteniendo matriz de puntos aleatorios...")

                                aleRSSIMatrix= getAlePointsMatrix(ale_points, ale_points_conf, beaconMacs, aleCampaignId, protocol, channel, rssiSamples)

                                updateTaskHistory(campaignId, taskId, "Aplicando método de posicionamiento...")

                                predicted_points = applyMethod(refRSSIMatrix, aleRSSIMatrix, method, metric, algorithm, ksRange, kernel, cs, gammas, nus, Is, taskId)
                                validateFiniteResults(predicted_points, 'Predicted_points')

                                mean_error = calculateError(aleRSSIMatrix, predicted_points)
                                validateFiniteResults(mean_error, 'Mean_error')

                                createMethodPrediction(
                                    campaignId,
                                    taskId,
                                    json.dumps(predicted_points, allow_nan=False),
                                    json.dumps(mean_error, allow_nan=False)
                                )

                                datetime_end = datetime.now()
                                updateTaskHistory(campaignId, taskId, "Completado", datetime_end)
                                completedTasks += 1

                            except Exception as e: 
                                failedTasks += 1
                                updateTaskHistory(campaignId, taskId, "Error")
                                print("Method processing error:", str(e))
                        else:
                            skippedTasks += 1

    return {
        'completedTasks': completedTasks,
        'skippedTasks': skippedTasks,
        'failedTasks': failedTasks
    }


def processDeepLearningHybrid(data):
    campaignId = data['campaign']
    campaignName = data['campaignName']
    protocols = data['protocols']
    channels = sorted(int(channel) for channel in data['channels'])
    rssiSamples = data['sample']
    if channels != [37, 38, 39]:
        raise ValueError('Hybrid requires channels 37, 38 and 39')

    configPoints = getPointsByCampaignId(campaignId)
    aleCampaignId = getRelatedCampaignIdByCampaignId(campaignId)
    refConfiguration = configPoints['Ref_points']
    aleConfiguration = configPoints['Ale_points']
    refPoints = getUniqueCoordinatesByCampaignId(
        campaignId, len(refConfiguration)
    )
    alePoints = getUniqueCoordinatesByCampaignId(
        aleCampaignId, len(aleConfiguration)
    )
    allowedRotations = {
        float(point['Rotation']) for point in aleConfiguration.values()
    }
    completedTasks = 0
    skippedTasks = 0
    failedTasks = 0

    for protocol in protocols:
        beaconMacs = processBeaconMacs(campaignId, aleCampaignId, protocol)
        if not beaconMacs:
            raise ValueError(f'No common beacons for protocol {protocol}')
        taskId = taskToJson(
            'DeepLearningHybrid', protocol, '37+38+39', rssiSamples,
            0, 0, '[0, 0]', 0, [None], [None], [None], [None], beaconMacs
        )
        if checkExistingMethodPrediction(campaignId, taskId):
            skippedTasks += 1
            continue
        try:
            createTaskHistory(
                campaignId, campaignName,
                'Obteniendo matrices multicanal de referencia...',
                taskId, datetime.now()
            )
            trainMatrices = [
                getDeepLearningRefPointsMatrix(
                    refPoints, refConfiguration, beaconMacs, campaignId,
                    protocol, channel, rssiSamples, allowedRotations
                ) for channel in channels
            ]
            updateTaskHistory(
                campaignId, taskId,
                'Obteniendo matrices multicanal de evaluaci?n...'
            )
            testMatrices = [
                getAlePointsMatrix(
                    alePoints, aleConfiguration, beaconMacs, aleCampaignId,
                    protocol, channel, rssiSamples
                ) for channel in channels
            ]
            updateTaskHistory(
                campaignId, taskId, 'Entrenando ensemble h?brido en GPU...'
            )
            predictedPoints = applyHybridDeepLearningMethod(
                trainMatrices, testMatrices, channels, taskId
            )
            validateFiniteResults(predictedPoints, 'Predicted_points')
            meanError = calculateError(testMatrices[0], predictedPoints)
            validateFiniteResults(meanError, 'Mean_error')
            createMethodPrediction(
                campaignId, taskId,
                json.dumps(predictedPoints, allow_nan=False),
                json.dumps(meanError, allow_nan=False)
            )
            updateTaskHistory(
                campaignId, taskId, 'Completado', datetime.now()
            )
            completedTasks += 1
        except Exception as error:
            failedTasks += 1
            updateTaskHistory(campaignId, taskId, 'Error')
            print('Hybrid method processing error:', str(error))

    return {
        'completedTasks': completedTasks,
        'skippedTasks': skippedTasks,
        'failedTasks': failedTasks,
    }

def getDeepLearningRefPointsMatrix(
        points, points_conf, beaconMacs, campaignId, protocol, channel,
        rssiSamples, allowedRotations=None):
    rows = []
    minimumObservedBeacons = max(1, len(beaconMacs) // 3)

    for point, point_values in zip(points, points_conf.values()):
        coordinates = np.array([
            float(point_values['X']),
            float(point_values['Y']),
            float(point_values['Z'])
        ])

        for rotation in point_values['Rotations']:
            if (
                allowedRotations is not None
                and float(rotation) not in allowedRotations
            ):
                continue
            samplesByBeacon = []

            for mac in beaconMacs:
                samples = getFilteredPredataSamples(
                    campaignId, 0, float(rotation), mac, protocol, channel,
                    point['x'], point['y'], point['z']
                )
                samplesByBeacon.append(samples)

            fingerprints = buildCaptureFingerprints(
                samplesByBeacon, rssiSamples, minimumObservedBeacons,
                DL_AGGREGATE_FINGERPRINT_REPETITIONS
            )
            for fingerprint in fingerprints:
                rows.append(np.concatenate((fingerprint, coordinates)))

    if len(rows) < 2:
        raise ValueError(
            'DeepLearning requires at least two reference fingerprints'
        )

    return np.asarray(rows, dtype=float)


def buildCaptureFingerprints(
        samplesByBeacon, rssiSamples, minimumObservedBeacons,
        aggregateRepetitions=1):
    captureValues = {}
    aggregateFingerprint = np.full(len(samplesByBeacon), MISSING_RSSI)

    for beaconIndex, samples in enumerate(samplesByBeacon):
        valuesByCapture = {}

        for sample in samples:
            valuesByCapture.setdefault(sample.Date, []).append(float(sample.RSSI))

        selectedValues = []
        for captureDate, values in valuesByCapture.items():
            if rssiSamples > 0:
                values = values[:rssiSamples]
            if not values:
                continue

            strongestValue = max(values)
            captureValues.setdefault(
                captureDate,
                np.full(len(samplesByBeacon), MISSING_RSSI)
            )[beaconIndex] = strongestValue
            selectedValues.extend(values)

        if selectedValues:
            aggregateFingerprint[beaconIndex] = max(selectedValues)

    fingerprints = [
        fingerprint
        for _, fingerprint in sorted(
            captureValues.items(), key=lambda item: str(item[0])
        )
        if np.count_nonzero(fingerprint > MISSING_RSSI)
        >= minimumObservedBeacons
    ]

    if (
        np.count_nonzero(aggregateFingerprint > MISSING_RSSI)
        >= minimumObservedBeacons
    ):
        fingerprints.extend(
            aggregateFingerprint.copy()
            for _ in range(aggregateRepetitions)
        )

    return fingerprints


def getRefPointsMatrix(points, points_conf, beaconMacs, campaignId, protocol, channel, rssiSamples):
    coordinates = 3
    cols = len(beaconMacs) + coordinates
    rows = len(points)

    refPointsMatrix = np.full((rows, cols), MISSING_RSSI)

    for i, (point, point_values) in enumerate(zip(points, points_conf.values())):
        x = float(point_values['X'])
        y = float(point_values['Y'])
        z = float(point_values['Z'])
        rotations = point_values['Rotations']

        for j, mac in enumerate(beaconMacs):
            maximumSamples = []

            for rotation in rotations:
                matchSamples = getFilteredPredataSamples(campaignId, rssiSamples, float(rotation), mac, protocol, channel, point['x'], point['y'], point['z'])

                if matchSamples:
                    maximumSample = max(matchSamples, key=lambda sample: sample.RSSI)
                    maximumSamples.append(maximumSample.RSSI)

            if maximumSamples:
                refPointsMatrix[i,j] = sum(maximumSamples) / len(maximumSamples)

        refPointsMatrix[i,cols - 3] = x
        refPointsMatrix[i,cols - 2] = y
        refPointsMatrix[i,cols - 1] = z

    return refPointsMatrix


def getAlePointsMatrix(points, points_conf, beaconMacs, campaignId, protocol, channel, rssiSamples):

    coordinates = 3
    cols = len(beaconMacs) + coordinates
    rows = len(points)

    alePointsMatrix = np.full((rows, cols), MISSING_RSSI)
    
    for i, (point, point_values) in enumerate(zip(points, points_conf.values())):
        x = float(point_values['X'])
        y = float(point_values['Y'])
        z = float(point_values['Z'])
        rotation = float(point_values['Rotation'])

        
        for j, mac in enumerate(beaconMacs):
            matchSamples = getFilteredPredataSamples(campaignId, rssiSamples, rotation, mac, protocol, channel, point['x'], point['y'], point['z'])

            if matchSamples:
                maximumSample = max(matchSamples, key=lambda sample: sample.RSSI)
                alePointsMatrix[i,j] = maximumSample.RSSI

        alePointsMatrix[i,cols - 3] = x
        alePointsMatrix[i,cols - 2] = y
        alePointsMatrix[i,cols - 1] = z

    return alePointsMatrix

def processBeaconMacs(referenceCampaignId, aleatoryCampaignId, protocol):
    aleatoryBeaconMacs = set(getObservedMacsByCampaignAndProtocol(
        aleatoryCampaignId, protocol
    ))
    referenceBeaconMacs = set(getObservedMacsByCampaignAndProtocol(
        referenceCampaignId, protocol
    ))

    return sorted(aleatoryBeaconMacs & referenceBeaconMacs)


def taskToJson(methods, protocols, channels, rssiSamples, algorithm, metric, ksRange, kernel, cs, gammas, nus, Is, beaconMacs=None):
    taskDescription = {
        'method': methods,
        'protocol': protocols,
        'channel': channels,
        'rssiSample': rssiSamples,
        'algorithm': algorithm,
        'metric': metric,
        'ksRange': ksRange,
        'kernel': kernel,
        'cs': cs,
        'gammas': gammas,
        'nus': nus,
        'Is': Is
    }

    if beaconMacs is not None:
        taskDescription['beaconCount'] = len(beaconMacs)
        taskDescription['beaconMacs'] = list(beaconMacs)

    if methods == 'DeepLearning':
        taskDescription['deepLearning'] = getDeepLearningConfiguration()
    elif methods == 'DeepLearningPyTorch':
        taskDescription['deepLearningPyTorch'] = (
            getPyTorchDeepLearningConfiguration()
        )

    if methods == 'DeepLearningHybrid':
        taskDescription['deepLearningHybrid'] = (
            getHybridDeepLearningConfiguration()
        )
    return json.dumps(taskDescription)
