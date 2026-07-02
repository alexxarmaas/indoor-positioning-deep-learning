import os

import numpy as np

from services.MethodsService import (
    DL_ENSEMBLE_SIZE,
    DL_MIN_SAMPLES_FOR_EARLY_STOPPING,
    DL_MODEL_CANDIDATES,
    DL_RANDOM_STATE,
    clipPredictionsToTrainingBounds,
    createDeepLearningRegressor,
    getKeyDescription,
    prepareDeepLearningFeatures,
)
from services.PyTorchPositioningService import (
    clipToTrainingBounds,
    getTorchDevice,
    predictTorchModel,
    trainTorchModel,
)

HYBRID_REQUIRED_CHANNELS = (37, 38, 39)
HYBRID_FUSION_WEIGHT = 0.85
HYBRID_LEGACY_WEIGHT = 0.15
HYBRID_TORCH_ENSEMBLE_SIZE = int(
    os.getenv('IPSCM_HYBRID_PRODUCTION_MODELS', '48')
)
HYBRID_TORCH_EPOCHS = int(
    os.getenv('IPSCM_HYBRID_PRODUCTION_EPOCHS', '1800')
)
HYBRID_TORCH_CONFIGURATIONS = (
    {
        'width': 48, 'blocks': 1, 'dropout': 0.22513071816875346,
        'learningRate': 7.114911501655661e-05,
        'weightDecay': 3.308940038882408e-05,
        'huberDelta': 0.7850938310724918, 'tailFraction': 0.2,
        'tailWeight': 0.5442650710200665, 'noiseStd': 0.12,
    },
    {
        'width': 24, 'blocks': 1, 'dropout': 0.2037008406914715,
        'learningRate': 0.0002195572647262685,
        'weightDecay': 7.536585776963143e-05,
        'huberDelta': 1.676808424905492, 'tailFraction': 0.25,
        'tailWeight': 0.7091718877129726, 'noiseStd': 0.02,
    },
    {
        'width': 48, 'blocks': 1, 'dropout': 0.3421024687833715,
        'learningRate': 0.00012858315183176592,
        'weightDecay': 5.778317230496835e-05,
        'huberDelta': 0.5712410504580595, 'tailFraction': 0.25,
        'tailWeight': 0.6312195516987211, 'noiseStd': 0.08,
    },
)
HYBRID_LEGACY_CONFIGURATIONS = (
    DL_MODEL_CANDIDATES[0],
    DL_MODEL_CANDIDATES[2],
)


def _predictLegacyEnsemble(trainFeatures, trainTargets, testFeatures):
    predictions = []
    useEarlyStopping = (
        len(trainFeatures) >= DL_MIN_SAMPLES_FOR_EARLY_STOPPING
    )
    for modelIndex in range(DL_ENSEMBLE_SIZE):
        configuration = HYBRID_LEGACY_CONFIGURATIONS[
            modelIndex % len(HYBRID_LEGACY_CONFIGURATIONS)
        ]
        model = createDeepLearningRegressor(
            randomState=DL_RANDOM_STATE + modelIndex,
            useEarlyStopping=useEarlyStopping,
            configuration=configuration,
        )
        model.fit(trainFeatures, trainTargets)
        predictions.append(clipPredictionsToTrainingBounds(
            model.predict(testFeatures), trainTargets
        ))
    return np.median(predictions, axis=0)


def applyHybridDeepLearningMethod(
        channelTrainMatrices, channelTestMatrices, channels, description):
    normalizedChannels = tuple(sorted(int(channel) for channel in channels))
    if normalizedChannels != HYBRID_REQUIRED_CHANNELS:
        raise ValueError('Hybrid requires channels 37, 38 and 39')
    if len(channelTrainMatrices) != len(HYBRID_REQUIRED_CHANNELS):
        raise ValueError('Missing hybrid reference channel matrix')
    if len(channelTestMatrices) != len(HYBRID_REQUIRED_CHANNELS):
        raise ValueError('Missing hybrid evaluation channel matrix')

    trainTargets = np.asarray(channelTrainMatrices[0][:, -3:], dtype=float)
    testTargets = np.asarray(channelTestMatrices[0][:, -3:], dtype=float)
    for matrix in channelTrainMatrices[1:]:
        if not np.allclose(trainTargets, matrix[:, -3:]):
            raise ValueError('Reference fingerprints are not channel-aligned')
    for matrix in channelTestMatrices[1:]:
        if not np.allclose(testTargets, matrix[:, -3:]):
            raise ValueError('Evaluation points are not channel-aligned')

    channelTrainFeatures = [
        prepareDeepLearningFeatures(matrix[:, :-3])
        for matrix in channelTrainMatrices
    ]
    channelTestFeatures = [
        prepareDeepLearningFeatures(matrix[:, :-3])
        for matrix in channelTestMatrices
    ]
    fusionTrainFeatures = np.concatenate(channelTrainFeatures, axis=1)
    fusionTestFeatures = np.concatenate(channelTestFeatures, axis=1)

    legacyPrediction = _predictLegacyEnsemble(
        channelTrainFeatures[0], trainTargets, channelTestFeatures[0]
    )
    fusionPredictions = []
    for modelIndex in range(HYBRID_TORCH_ENSEMBLE_SIZE):
        configuration = HYBRID_TORCH_CONFIGURATIONS[
            modelIndex % len(HYBRID_TORCH_CONFIGURATIONS)
        ]
        trainingResult = trainTorchModel(
            fusionTrainFeatures, trainTargets, configuration,
            50000 + modelIndex, HYBRID_TORCH_EPOCHS,
        )
        prediction = predictTorchModel(trainingResult, fusionTestFeatures)
        fusionPredictions.append(clipToTrainingBounds(
            prediction, trainTargets
        ))
    fusionPrediction = np.median(fusionPredictions, axis=0)
    hybridPrediction = (
        HYBRID_FUSION_WEIGHT * fusionPrediction
        + HYBRID_LEGACY_WEIGHT * legacyPrediction
    )
    hybridPrediction = clipPredictionsToTrainingBounds(
        hybridPrediction, trainTargets
    )

    keyName = getKeyDescription(description)
    resultKey = (
        keyName + '[model=Hybrid-Sklearn-PyTorch]'
        + '[channels=37+38+39]'
        + f'[weights={HYBRID_FUSION_WEIGHT:.2f}+'
        + f'{HYBRID_LEGACY_WEIGHT:.2f}]'
        + f'[ensemble={HYBRID_TORCH_ENSEMBLE_SIZE}+{DL_ENSEMBLE_SIZE}]'
    )
    return {resultKey: hybridPrediction.tolist()}


def getHybridDeepLearningConfiguration():
    return {
        'frameworks': ['PyTorch', 'scikit-learn'],
        'channels': list(HYBRID_REQUIRED_CHANNELS),
        'fusionWeight': HYBRID_FUSION_WEIGHT,
        'legacyChannel37Weight': HYBRID_LEGACY_WEIGHT,
        'torchArchitecture': 'ResidualMLP',
        'torchConfigurations': list(HYBRID_TORCH_CONFIGURATIONS),
        'torchEnsembleSize': HYBRID_TORCH_ENSEMBLE_SIZE,
        'torchEpochs': HYBRID_TORCH_EPOCHS,
        'legacyEnsembleSize': DL_ENSEMBLE_SIZE,
        'selection': 'grouped 5-fold OOF on reference fingerprints',
        'aggregation': 'coordinate-wise median then weighted mean',
        'device': str(getTorchDevice()),
    }
