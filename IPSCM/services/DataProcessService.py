import json
import numpy as np
from datetime import datetime
from controllers.BeaconBleSignalCrud import (
    countBeaconBleSignalsByCaptureIds, getBeaconBleSignalById,
    getObservedMacsByCampaignAndProtocol
)
from controllers.CampaignCrud import getPointsByCampaignId, getRelatedCampaignIdByCampaignId
from controllers.CaptureCrud import getCaptureIdsByCampaignId, getCapturesById
from controllers.MethodPredataCrud import (
    createMethodPredataBatch,
    getFilteredPredataSamples,
    getUniqueCoordinatesByCampaignId,
    hasPredataForFingerprint,
)
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
from services.TFGFeaturesService import build_tfg_campaign_matrix, load_beacon_mapping
from services.TFGPositioningService import TFG_METHODS, TFG_MLP_CONFIG


ORIENTATION_PROCESSING_VERSION = 3

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
    if method in TFG_METHODS:
        return processTFGMethod(data)
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
    trainingRotations = getConfiguredRotations(ref_points_conf)
    evaluationRotations = getConfiguredRotations(ale_points_conf)

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
        evaluationSamplePlan = buildEvaluationSamplePlan(
            ale_points, ale_points_conf, beaconMacs, aleCampaignId, protocol
        )
        orientationHandling = buildOrientationHandling(
            trainingRotations, evaluationRotations,
            len(evaluationSamplePlan),
            countConfiguredSamples(ale_points_conf),
        )
        for channel in channels:
            for metric in metrics:
                for algorithm in algorithms:
                    for kernel in kernels:

                        taskId = taskToJson(
                            method, protocol, channel, rssiSamples, algorithm,
                            metric, ksRangeJson, kernel, cs, gammas, nus, Is,
                            beaconMacs, orientationHandling
                        )
                        
                        if not checkExistingMethodPrediction(campaignId, taskId):

                            try:
                                datetime_start = datetime.now()
                                createTaskHistory(campaignId, campaignName, "Obteniendo matriz de puntos de referencia...", taskId, datetime_start)
                                
                                if method in (
                                    'DeepLearning',
                                    'DeepLearningPyTorch',
                                ):
                                    refRSSIMatrix = getDeepLearningRefPointsMatrix(
                                        ref_points, ref_points_conf, beaconMacs,
                                        campaignId, protocol, channel, rssiSamples,
                                        set(trainingRotations)
                                    )
                                else:
                                    refRSSIMatrix = getRefPointsMatrix(
                                        ref_points, ref_points_conf, beaconMacs,
                                        campaignId, protocol, channel, rssiSamples
                                    )

                                updateTaskHistory(campaignId, taskId, "Obteniendo matriz de puntos aleatorios...")

                                aleRSSIMatrix = getAlePointsMatrix(
                                    ale_points, ale_points_conf, beaconMacs,
                                    aleCampaignId, protocol, channel,
                                    rssiSamples, evaluationSamplePlan
                                )

                                updateTaskHistory(campaignId, taskId, "Aplicando método de posicionamiento...")

                                predicted_points = applyMethod(refRSSIMatrix, aleRSSIMatrix, method, metric, algorithm, ksRange, kernel, cs, gammas, nus, Is, taskId)
                                validateFiniteResults(predicted_points, 'Predicted_points')

                                fusedAleMatrix, predicted_points = (
                                    fusePredictionsByPointMedian(
                                        aleRSSIMatrix, predicted_points,
                                        evaluationSamplePlan
                                    )
                                )
                                validateFiniteResults(
                                    predicted_points, 'Fused predicted_points'
                                )
                                mean_error = calculateError(
                                    fusedAleMatrix, predicted_points
                                )
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
    trainingRotations = getConfiguredRotations(refConfiguration)
    evaluationRotations = getConfiguredRotations(aleConfiguration)
    completedTasks = 0
    skippedTasks = 0
    failedTasks = 0

    for protocol in protocols:
        beaconMacs = processBeaconMacs(campaignId, aleCampaignId, protocol)
        if not beaconMacs:
            raise ValueError(f'No common beacons for protocol {protocol}')
        evaluationSamplePlan = buildEvaluationSamplePlan(
            alePoints, aleConfiguration, beaconMacs, aleCampaignId, protocol
        )
        orientationHandling = buildOrientationHandling(
            trainingRotations, evaluationRotations,
            len(evaluationSamplePlan),
            countConfiguredSamples(aleConfiguration),
        )
        taskId = taskToJson(
            'DeepLearningHybrid', protocol, '37+38+39', rssiSamples,
            0, 0, '[0, 0]', 0, [None], [None], [None], [None],
            beaconMacs, orientationHandling
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
                    protocol, channel, rssiSamples, set(trainingRotations)
                ) for channel in channels
            ]
            updateTaskHistory(
                campaignId, taskId,
                'Obteniendo matrices multicanal de evaluaci?n...'
            )
            testMatrices = [
                getAlePointsMatrix(
                    alePoints, aleConfiguration, beaconMacs, aleCampaignId,
                    protocol, channel, rssiSamples, evaluationSamplePlan
                ) for channel in channels
            ]
            updateTaskHistory(
                campaignId, taskId, 'Entrenando ensemble h?brido en GPU...'
            )
            predictedPoints = applyHybridDeepLearningMethod(
                trainMatrices, testMatrices, channels, taskId
            )
            validateFiniteResults(predictedPoints, 'Predicted_points')
            fusedTestMatrix, predictedPoints = (
                fusePredictionsByPointMedian(
                    testMatrices[0], predictedPoints, evaluationSamplePlan
                )
            )
            validateFiniteResults(
                predictedPoints, 'Fused predicted_points'
            )
            meanError = calculateError(fusedTestMatrix, predictedPoints)
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
        rotations = getPointRotations(point_values)

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


def getPointRotations(pointValues):
    rotations = pointValues.get('Rotations')
    if rotations is None:
        rotations = [pointValues['Rotation']]

    uniqueRotations = []
    for rotation in rotations:
        numericRotation = float(rotation)
        if numericRotation not in uniqueRotations:
            uniqueRotations.append(numericRotation)

    if not uniqueRotations:
        raise ValueError('At least one orientation must be configured')
    return uniqueRotations


def getConfiguredRotations(pointsConfiguration):
    rotations = {
        rotation
        for pointValues in pointsConfiguration.values()
        for rotation in getPointRotations(pointValues)
    }
    if not rotations:
        raise ValueError('No orientations found in point configuration')
    return sorted(rotations)


def countConfiguredSamples(pointsConfiguration):
    return sum(
        len(getPointRotations(pointValues))
        for pointValues in pointsConfiguration.values()
    )


def buildEvaluationSamplePlan(
        points, pointsConfiguration, beaconMacs, campaignId, protocol):
    samplePlan = []
    for pointIndex, (point, pointValues) in enumerate(
            zip(points, pointsConfiguration.values())):
        for rotation in getPointRotations(pointValues):
            if hasPredataForFingerprint(
                campaignId, rotation, beaconMacs, protocol,
                point['x'], point['y'], point['z']
            ):
                samplePlan.append((pointIndex, rotation))

    if not samplePlan:
        raise ValueError(
            'No valid evaluation captures were found for the configured '
            'orientations'
        )
    return samplePlan


def buildOrientationHandling(
        trainingRotations, evaluationRotations, validSamples, totalSamples):
    return {
        'version': ORIENTATION_PROCESSING_VERSION,
        'mode': 'multi-orientation-median-fusion',
        'trainingRotations': list(trainingRotations),
        'evaluationRotations': list(evaluationRotations),
        'evaluationAggregation': 'coordinate-wise median prediction per point',
        'generalResultUnit': 'physical evaluation point',
        'validEvaluationCaptures': validSamples,
        'configuredEvaluationCaptures': totalSamples,
        'excludedEmptyEvaluationCaptures': totalSamples - validSamples,
    }


def fusePredictionsByPointMedian(
        evaluationMatrix, predictedPoints, evaluationSamplePlan):
    evaluationMatrix = np.asarray(evaluationMatrix, dtype=float)
    if len(evaluationMatrix) != len(evaluationSamplePlan):
        raise ValueError(
            'Evaluation matrix and orientation sample plan are not aligned'
        )

    rowIndexesByPoint = {}
    for rowIndex, (pointIndex, _) in enumerate(evaluationSamplePlan):
        rowIndexesByPoint.setdefault(pointIndex, []).append(rowIndex)

    fusedTargets = []
    for rowIndexes in rowIndexesByPoint.values():
        targets = evaluationMatrix[rowIndexes, -3:]
        if not np.allclose(targets, targets[0]):
            raise ValueError(
                'Orientation samples for a point have different targets'
            )
        fusedTargets.append(targets[0])

    fusedPredictions = {}
    for key, values in predictedPoints.items():
        predictions = np.asarray(values, dtype=float)
        if len(predictions) != len(evaluationSamplePlan):
            raise ValueError(
                'Predictions and orientation sample plan are not aligned'
            )
        if predictions.ndim != 2 or predictions.shape[1] != 3:
            raise ValueError('Predictions must contain x, y and z coordinates')

        fusedKey = f'{key}[orientationFusion=coordinate-median]'
        fusedPredictions[fusedKey] = [
            np.median(predictions[rowIndexes], axis=0).tolist()
            for rowIndexes in rowIndexesByPoint.values()
        ]

    return np.asarray(fusedTargets, dtype=float), fusedPredictions


def getAlePointsMatrix(
        points, points_conf, beaconMacs, campaignId, protocol, channel,
        rssiSamples, evaluationSamplePlan=None):
    coordinates = 3
    cols = len(beaconMacs) + coordinates
    configurationValues = list(points_conf.values())
    if evaluationSamplePlan is None:
        evaluationSamplePlan = [
            (pointIndex, rotation)
            for pointIndex, pointValues in enumerate(configurationValues)
            for rotation in getPointRotations(pointValues)
        ]

    rows = []
    for pointIndex, rotation in evaluationSamplePlan:
        point = points[pointIndex]
        point_values = configurationValues[pointIndex]
        x = float(point_values['X'])
        y = float(point_values['Y'])
        z = float(point_values['Z'])
        row = np.full(cols, MISSING_RSSI)
        for j, mac in enumerate(beaconMacs):
            matchSamples = getFilteredPredataSamples(campaignId, rssiSamples, rotation, mac, protocol, channel, point['x'], point['y'], point['z'])

            if matchSamples:
                maximumSample = max(matchSamples, key=lambda sample: sample.RSSI)
                row[j] = maximumSample.RSSI

        row[cols - 3] = x
        row[cols - 2] = y
        row[cols - 1] = z
        rows.append(row)

    if not rows:
        raise ValueError('Evaluation matrix is empty')

    return np.asarray(rows, dtype=float)

def processBeaconMacs(referenceCampaignId, aleatoryCampaignId, protocol):
    aleatoryBeaconMacs = set(getObservedMacsByCampaignAndProtocol(
        aleatoryCampaignId, protocol
    ))
    referenceBeaconMacs = set(getObservedMacsByCampaignAndProtocol(
        referenceCampaignId, protocol
    ))

    return sorted(aleatoryBeaconMacs & referenceBeaconMacs)


def taskToJson(
        methods, protocols, channels, rssiSamples, algorithm, metric, ksRange,
        kernel, cs, gammas, nus, Is, beaconMacs=None,
        orientationHandling=None):
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
    if orientationHandling is not None:
        taskDescription['orientationHandling'] = orientationHandling

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


def processTFGMethod(data):
    """Comparación TFG con huellas por captura, independientes de los canales UI.

    Este camino se mantiene separado del pipeline existente para no modificar
    WKNN, SVR, DeepLearning o los experimentos multicanal del fork.
    """
    campaign_id = int(data["campaign"])
    campaign_name = data["campaignName"]
    method = data["method"]
    evaluation_id = getRelatedCampaignIdByCampaignId(campaign_id)
    if campaign_id == evaluation_id:
        raise ValueError("Referencia y evaluación no pueden ser la misma campaña")

    train, train_ids = build_tfg_campaign_matrix(campaign_id)
    test, test_ids = build_tfg_campaign_matrix(evaluation_id)
    if set(train_ids).intersection(test_ids):
        raise ValueError("Fuga de datos: capturas compartidas entre entrenamiento y evaluación")

    coordinate_key = lambda row: tuple(np.round(row[-3:], 5))
    shared_coords = len({coordinate_key(row) for row in train}.intersection(
        {coordinate_key(row) for row in test}
    ))
    point_indices = {}
    evaluation_plan = []
    for row in test:
        key = coordinate_key(row)
        if key not in point_indices:
            point_indices[key] = len(point_indices)
        evaluation_plan.append((point_indices[key], None))

    if method == "WKNN_TFG":
        metrics = data.get("metrics") or ["euclidean"]
        algorithms = data.get("algorithms") or ["brute"]
        k_start = int(data.get("kRangeStart") or 1)
        k_end = min(int(data.get("kRangeEnd") or k_start), len(train))
        if k_start < 1 or k_end < k_start:
            raise ValueError("Rango de k incompatible con el número de capturas de referencia")
    else:
        metrics, algorithms, k_start, k_end = [0], [0], 0, 0

    mapping = load_beacon_mapping()
    completed, skipped, failed = 0, 0, 0
    failures = []

    for metric in metrics:
        for algorithm in algorithms:
            description = {
                "method": method,
                "protocol": "ALL",
                "channel": "ALL",
                "rssiSample": -1,
                "algorithm": algorithm,
                "metric": metric,
                "ksRange": json.dumps([k_start, k_end]),
                "tfgFeatures": {
                    "version": 1,
                    "unit": "capture",
                    "rssiAggregation": "median-by-MAC-across-all-channels",
                    "missingRSSI": -100,
                    "target": "xy-with-constant-z",
                    "evaluationAggregation": "median-prediction-per-physical-point",
                    "beaconMapping": mapping,
                    "referenceCampaignId": campaign_id,
                    "evaluationCampaignId": evaluation_id,
                    "nReferenceCaptures": len(train),
                    "nEvaluationCaptures": len(test),
                    "sharedReferenceEvaluationCoordinates": shared_coords,
                    "modelConfig": TFG_MLP_CONFIG if method != "WKNN_TFG" else None,
                },
            }
            task_id = json.dumps(description, sort_keys=True)
            if checkExistingMethodPrediction(campaign_id, task_id):
                skipped += 1
                continue
            started_at = datetime.now()
            try:
                createTaskHistory(
                    campaign_id, campaign_name,
                    "Evaluando fingerprints por captura (TFG)...",
                    task_id, started_at
                )
                predicted = applyMethod(
                    train, test, method, metric, algorithm,
                    [k_start, k_end], 0, [None], [None], [None], [None],
                    task_id
                )
                validateFiniteResults(predicted, "TFG predicted points")
                targets, fused = fusePredictionsByPointMedian(
                    test, predicted, evaluation_plan
                )
                validateFiniteResults(fused, "TFG fused predicted points")
                errors = calculateError(targets, fused)
                validateFiniteResults(errors, "TFG errors")
                createMethodPrediction(
                    campaign_id, task_id,
                    json.dumps(fused, allow_nan=False),
                    json.dumps(errors, allow_nan=False)
                )
                updateTaskHistory(campaign_id, task_id, "Completado", datetime.now())
                completed += 1
            except Exception as exc:
                failed += 1
                failures.append(str(exc))
                updateTaskHistory(campaign_id, task_id, "Error")
                print("TFG method processing error:", str(exc))

    return {
        "completedTasks": completed,
        "skippedTasks": skipped,
        "failedTasks": failed,
        "errors": failures,
        "referenceCaptures": len(train),
        "evaluationCaptures": len(test),
        "evaluationPhysicalPoints": len(point_indices),
        "overlappingCoordinates": shared_coords,
    }
