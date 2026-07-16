import json
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from services.PyTorchPositioningService import (
    getPyTorchConfiguration,
    predictWithPyTorchEnsemble,
)


DL_MODEL_CANDIDATES = (
    {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'alpha': 0.001,
    },
    {
        'hidden_layer_sizes': (256, 128, 64),
        'activation': 'relu',
        'alpha': 0.001,
    },
    {
        'hidden_layer_sizes': (128, 128, 64),
        'activation': 'tanh',
        'alpha': 0.003,
    },
)
DL_HIDDEN_LAYER_SIZES = DL_MODEL_CANDIDATES[0]['hidden_layer_sizes']
DL_ACTIVATION = DL_MODEL_CANDIDATES[0]['activation']
DL_ALPHA = DL_MODEL_CANDIDATES[0]['alpha']
DL_MAX_ITERATIONS = 2500
DL_RANDOM_STATE = 45
DL_ENSEMBLE_SIZE = 6
DL_TOP_CONFIGURATIONS = 2
DL_MODEL_SELECTION_FOLDS = 4
DL_MODEL_SELECTION_MIN_GROUPS = 6
DL_P90_WEIGHT = 0.5
DL_MIN_SAMPLES_FOR_EARLY_STOPPING = 100
DL_AGGREGATE_FINGERPRINT_REPETITIONS = 6
MISSING_RSSI = -110.0


def applyMethod(data_train, data_test, method, metrics, algorithms, ks_range, kernel, cs, gammas, nus, Is, description):
    match method:
        case "WKNN":
            return applyWKNNmethod(data_train, data_test, metrics, algorithms, ks_range, description)
        case "DeepLearning":
            return applyDeepLearningMethod(data_train, data_test, description)
        case 'DeepLearningPyTorch':
            return applyPyTorchDeepLearningMethod(
                data_train, data_test, description
            )
        case "SVR":
            return applySVRmethod(data_train, data_test, kernel, cs, gammas, description)
        case "NuSVR":
            return applyNuSVRmethod(data_train, data_test, kernel, cs, gammas, nus, description)
        case "LinearSVR":
            return applyLinearSVRmethod(data_train, data_test, Is, cs, description)


def inverseDistanceWeights(distances):
    distances = np.asarray(distances, dtype=float)
    weights = np.zeros_like(distances, dtype=float)

    if distances.ndim == 1:
        zeroDistances = distances == 0
        if zeroDistances.any():
            weights[zeroDistances] = 1.0
        else:
            weights = 1.0 / distances
        return weights

    zeroDistances = distances == 0
    rowsWithZero = zeroDistances.any(axis=1)
    if rowsWithZero.any():
        weights[rowsWithZero] = zeroDistances[rowsWithZero].astype(float)

    rowsWithoutZero = ~rowsWithZero
    if rowsWithoutZero.any():
        weights[rowsWithoutZero] = 1.0 / distances[rowsWithoutZero]

    return weights


def validateFiniteResults(results, resultName):
    invalidSeries = []

    for key, values in results.items():
        valuesArray = np.asarray(values, dtype=float)
        finiteMask = np.isfinite(valuesArray)
        if not finiteMask.all():
            invalidCount = valuesArray.size - np.count_nonzero(finiteMask)
            invalidSeries.append(f'{key}: {invalidCount} invalid values')

    if invalidSeries:
        details = '; '.join(invalidSeries[:5])
        raise ValueError(f'{resultName} contains NaN or infinite values: {details}')


def applyWKNNmethod(data_train, data_test, metric, algorithm, ks_range, description):
    keyName = getKeyDescription(description)

    X_train = prepareRSSIFeatures(data_train[:, :-3])
    y_train = data_train[:, -3:]

    results = {} 

    for i in range(ks_range[0], ks_range[1] + 1):
        regressor = KNeighborsRegressor(n_neighbors=i, weights=inverseDistanceWeights,
                                        metric=metric, algorithm=algorithm)
        regressor.fit(X_train, y_train)

        X_test = prepareRSSIFeatures(data_test[:, :-3])
        y_pred = regressor.predict(X_test).tolist()
        newKeyName = keyName + f'[k={str(i)}]'
        results[newKeyName] = y_pred

    return results


def applyDeepLearningMethod(data_train, data_test, description):
    keyName = getKeyDescription(description)

    X_train = prepareDeepLearningFeatures(data_train[:, :-3])
    y_train = data_train[:, -3:]
    X_test = prepareDeepLearningFeatures(data_test[:, :-3])

    if X_train.shape[0] < 2:
        raise ValueError("DeepLearning requires at least two reference points")
    if X_train.shape[1] == 0:
        raise ValueError("DeepLearning requires at least one common beacon")

    selectedConfigurations = selectDeepLearningConfigurations(X_train, y_train)
    predictions = []
    useEarlyStopping = X_train.shape[0] >= DL_MIN_SAMPLES_FOR_EARLY_STOPPING
    for ensembleIndex in range(DL_ENSEMBLE_SIZE):
        configuration = selectedConfigurations[
            ensembleIndex % len(selectedConfigurations)
        ]
        model = createDeepLearningRegressor(
            randomState=DL_RANDOM_STATE + ensembleIndex,
            useEarlyStopping=useEarlyStopping,
            configuration=configuration,
        )
        model.fit(X_train, y_train)
        predictions.append(clipPredictionsToTrainingBounds(
            model.predict(X_test), y_train
        ))

    y_pred = np.median(predictions, axis=0).tolist()
    layers = '+'.join(
        'x'.join(str(size) for size in item['hidden_layer_sizes'])
        for item in selectedConfigurations
    )
    results = {
        keyName + f'[layers={layers}][ensemble={DL_ENSEMBLE_SIZE}]': y_pred
    }

    return results


def applyPyTorchDeepLearningMethod(data_train, data_test, description):
    keyName = getKeyDescription(description)
    X_train = prepareDeepLearningFeatures(data_train[:, :-3])
    y_train = data_train[:, -3:]
    X_test = prepareDeepLearningFeatures(data_test[:, :-3])

    if X_train.shape[0] < 2:
        raise ValueError(
            'DeepLearningPyTorch requires at least two reference points'
        )
    if X_train.shape[1] == 0:
        raise ValueError(
            'DeepLearningPyTorch requires at least one common beacon'
        )

    prediction, selectedConfigurations = predictWithPyTorchEnsemble(
        X_train, y_train, X_test
    )
    architectures = '+'.join(
        'w{}b{}'.format(item['width'], item['blocks'])
        for item in selectedConfigurations
    )
    torchConfiguration = getPyTorchConfiguration()
    resultKey = (
        keyName
        + '[model=PyTorch-ResMLP]'
        + '[architectures=' + architectures + ']'
        + '[ensemble=' + str(torchConfiguration['ensembleSize']) + ']'
    )
    return {resultKey: prediction.tolist()}


def createDeepLearningRegressor(
        randomState,
        useEarlyStopping,
        configuration=None):
    configuration = configuration or DL_MODEL_CANDIDATES[0]

    if useEarlyStopping:
        solver = 'adam'
        trainingOptions = {
            'batch_size': 'auto',
            'early_stopping': True,
            'validation_fraction': 0.15,
            'n_iter_no_change': 80,
            'learning_rate_init': 0.0007,
        }
    else:
        solver = 'lbfgs'
        trainingOptions = {
            'max_fun': 30000,
        }

    return TransformedTargetRegressor(
        regressor=Pipeline([
            ('input_scaler', StandardScaler()),
            ('neural_network', MLPRegressor(
                hidden_layer_sizes=configuration['hidden_layer_sizes'],
                activation=configuration['activation'],
                solver=solver,
                alpha=configuration['alpha'],
                max_iter=DL_MAX_ITERATIONS,
                tol=1e-5,
                random_state=randomState,
                **trainingOptions
            ))
        ]),
        transformer=StandardScaler()
    )


def selectDeepLearningConfigurations(features, targets):
    _, groups = np.unique(
        np.round(np.asarray(targets, dtype=float), 6),
        axis=0,
        return_inverse=True,
    )
    groupCount = len(np.unique(groups))
    if groupCount < DL_MODEL_SELECTION_MIN_GROUPS:
        return [DL_MODEL_CANDIDATES[0]]
    splitter = GroupKFold(
        n_splits=min(DL_MODEL_SELECTION_FOLDS, groupCount)
    )
    scores = [
        (
            scoreDeepLearningConfiguration(
                candidate, features, targets, groups, splitter, index
            ),
            candidate,
        )
        for index, candidate in enumerate(DL_MODEL_CANDIDATES)
    ]
    scores.sort(key=lambda item: item[0])
    return [candidate for _, candidate in scores[:DL_TOP_CONFIGURATIONS]]


def scoreDeepLearningConfiguration(
        configuration, features, targets, groups, splitter, candidateIndex):
    foldScores = []
    for trainIndexes, validationIndexes in splitter.split(
            features, targets, groups):
        model = createDeepLearningRegressor(
            DL_RANDOM_STATE + candidateIndex,
            len(trainIndexes) >= DL_MIN_SAMPLES_FOR_EARLY_STOPPING,
            configuration,
        )
        model.fit(features[trainIndexes], targets[trainIndexes])
        prediction = clipPredictionsToTrainingBounds(
            model.predict(features[validationIndexes]),
            targets[trainIndexes],
        )
        error = np.linalg.norm(
            targets[validationIndexes] - prediction,
            axis=1,
        )
        foldScores.append(
            np.mean(error) + DL_P90_WEIGHT * np.percentile(error, 90)
        )
    return np.mean(foldScores)


def clipPredictionsToTrainingBounds(predictions, targets):
    targets = np.asarray(targets, dtype=float)
    return np.clip(
        np.asarray(predictions, dtype=float),
        np.min(targets, axis=0),
        np.max(targets, axis=0),
    )


def prepareRSSIFeatures(features):
    features = np.asarray(features, dtype=float)
    return np.nan_to_num(
        features,
        copy=True,
        nan=MISSING_RSSI,
        neginf=MISSING_RSSI,
        posinf=MISSING_RSSI
    )


def prepareDeepLearningFeatures(features):
    rawFeatures = np.asarray(features, dtype=float)
    missingMask = (~np.isfinite(rawFeatures)) | (rawFeatures <= MISSING_RSSI)
    preparedFeatures = prepareRSSIFeatures(rawFeatures)
    observedValues = np.where(missingMask, 0.0, preparedFeatures)
    observedCounts = np.sum(~missingMask, axis=1, keepdims=True)
    observedSums = np.sum(observedValues, axis=1, keepdims=True)
    rowMeans = np.divide(
        observedSums,
        observedCounts,
        out=np.full_like(observedSums, MISSING_RSSI),
        where=observedCounts > 0,
    )
    relativeFeatures = np.where(
        missingMask,
        0.0,
        preparedFeatures - rowMeans,
    )
    return np.concatenate(
        (preparedFeatures, relativeFeatures, missingMask.astype(float)),
        axis=1,
    )


def getDeepLearningConfiguration():
    return {
        'candidateModels': [
            {
                'hiddenLayers': list(candidate['hidden_layer_sizes']),
                'activation': candidate['activation'],
                'alpha': candidate['alpha'],
            }
            for candidate in DL_MODEL_CANDIDATES
        ],
        'selectionFolds': DL_MODEL_SELECTION_FOLDS,
        'selectionObjective': f'mean + {DL_P90_WEIGHT} * P90',
        'selectedModels': DL_TOP_CONFIGURATIONS,
        'solver': 'adam/lbfgs',
        'maxIterations': DL_MAX_ITERATIONS,
        'randomState': DL_RANDOM_STATE,
        'ensembleSize': DL_ENSEMBLE_SIZE,
        'ensembleAggregation': 'coordinate-wise median',
        'featureEngineering': ['absolute RSSI', 'relative RSSI', 'missing mask'],
        'spatialClipping': True,
        'trainingSamples': 'captures matching aleatory rotations',
        'aggregateFingerprintRepetitions':
            DL_AGGREGATE_FINGERPRINT_REPETITIONS,
        'missingRSSI': MISSING_RSSI
    }


def getPyTorchDeepLearningConfiguration():
    configuration = getPyTorchConfiguration()
    configuration.update({
        'featureEngineering': [
            'absolute RSSI',
            'relative RSSI',
            'missing mask',
        ],
        'spatialClipping': True,
        'trainingSamples': 'captures matching aleatory rotations',
        'aggregateFingerprintRepetitions':
            DL_AGGREGATE_FINGERPRINT_REPETITIONS,
        'missingRSSI': MISSING_RSSI,
    })
    return configuration


def applySVRmethod(data_train, data_test, kernel, cs, gammas, description):
    keyName = getKeyDescription(description)

    X_train = data_train[:, :-3]
    y_train = data_train[:, -3:]

    results = {} 

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(data_test[:, :-3])

    for c in cs:
        if isinstance(gammas, list):
            for gamma in gammas:
                model = SVR(kernel=kernel, C=c, gamma=gamma)
                wrapper = MultiOutputRegressor(model).fit(X_train, y_train)

                y_pred = wrapper.predict(X_test)
                y_pred_list = y_pred.tolist() 

                newKeyName = keyName + f'[k={f"C{c}_G{gamma}"}]'
                results[newKeyName] = y_pred_list
        else:
            model = SVR(kernel=kernel, C=c, gamma=gammas)
            wrapper = MultiOutputRegressor(model).fit(X_train, y_train)

            y_pred = wrapper.predict(X_test)
            y_pred_list = y_pred.tolist() 

            newKeyName = keyName + f'[k={f"C{c}_G{gammas}"}]'
            results[newKeyName] = y_pred_list

    return results


def applyNuSVRmethod(data_train, data_test, kernel, cs, gammas, nus, description):
    keyName = getKeyDescription(description)

    X_train = data_train[:, :-3]
    y_train = data_train[:, -3:]

    results = {} 

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(data_test[:, :-3])

    for nu in nus:
        for c in cs:
            if isinstance(gammas, list):
                for gamma in gammas:
                    model = NuSVR(nu=nu, kernel=kernel, C=c, gamma=gamma)
                    wrapper = MultiOutputRegressor(model).fit(X_train, y_train)

                    y_pred = wrapper.predict(X_test)
                    y_pred_list = y_pred.tolist()

                    newKeyName = keyName + f'[k={f"nu{nu}_C{c}_G{gamma}"}]'
                    results[newKeyName] = y_pred_list
            else:
                model = NuSVR(nu=nu, kernel=kernel, C=c, gamma=gammas)
                wrapper = MultiOutputRegressor(model).fit(X_train, y_train)

                y_pred = wrapper.predict(X_test)
                y_pred_list = y_pred.tolist()
                newKeyName = keyName + f'[k={f"nu{nu}_C{c}_G{gammas}"}]'
                results[newKeyName] = y_pred_list

    return results
                   
def applyLinearSVRmethod(data_train, data_test, Is, cs, description):
    keyName = getKeyDescription(description)

    X_train = data_train[:, :-3]
    y_train = data_train[:, -3:]

    results = {} 

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(data_test[:, :-3])


    for i in Is:
        for c in cs:
                model = LinearSVR(C=c, max_iter=i, random_state=0)
                wrapper = MultiOutputRegressor(model).fit(X_train, y_train)

                y_pred = wrapper.predict(X_test)
                y_pred_list = y_pred.tolist()

                newKeyName = keyName + f'[k={f"i{i}_C{c}"}]'
                results[newKeyName] = y_pred_list

    return results

def calculateError(aleMatrix, results):

    y_test = aleMatrix[:, -3:]
    errors = {}

    for k, y_pred in results.items():   
        error = np.sqrt(np.sum((y_test - y_pred) ** 2, axis=1))
        mean_error_rounded = round(np.mean(error), 3)
        newKeyName = f'{str(k)}[Mean={str(mean_error_rounded)}]'
        errors[newKeyName] = error

    errorsList = {k: sorted(v) for k, v in errors.items()}
        
    return errorsList


def getKeyDescription(description):
    key = json.loads(description)
    keyName = ''
    excludedFields = {
        'ksRange', 'cs', 'Is', 'nus', 'gammas', 'deepLearning',
        'deepLearningPyTorch', 'deepLearningHybrid', 'beaconCount',
        'beaconMacs',
    }

    for field, value in key.items():
        if value != [None] and value != 0 and field not in excludedFields:
            keyName += f'[{value}]'
    
    return keyName
