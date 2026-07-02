import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import app
from experiments.optimize_pytorch import (
    OUTPUT_DIR,
    loadFusionMatrix,
    loadReferenceMatrix,
    positionGroups,
)
from services.MethodsService import (
    DL_MIN_SAMPLES_FOR_EARLY_STOPPING,
    clipPredictionsToTrainingBounds,
    createDeepLearningRegressor,
)
import services.PyTorchPositioningService as torchService


FOLDS = 5
FOLD_INDEX = int(os.getenv('IPSCM_HYBRID_FOLD', '0'))
TORCH_MODELS = int(os.getenv('IPSCM_HYBRID_MODELS', '12'))
TORCH_EPOCHS = int(os.getenv('IPSCM_HYBRID_EPOCHS', '1200'))
SEED_BASE = 70000

LEGACY_CONFIGURATIONS = (
    {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'alpha': 0.001,
    },
    {
        'hidden_layer_sizes': (128, 128, 64),
        'activation': 'tanh',
        'alpha': 0.003,
    },
)


def loadFusionConfigurations():
    path = OUTPUT_DIR / 'pytorch_confirmation_fusion_37_38_39.json'
    confirmation = json.loads(path.read_text(encoding='utf-8'))
    summaries = [
        item for item in confirmation['summaries']
        if item['completedRepeats'] == confirmation['repeats']
    ]
    summaries.sort(key=lambda item: item['meanScore'] + item['stdScore'])
    return [item['configuration'] for item in summaries[:3]]


def trainLegacyExpert(trainX, trainY, validationX):
    predictions = []
    useEarlyStopping = len(trainX) >= DL_MIN_SAMPLES_FOR_EARLY_STOPPING
    for modelIndex in range(6):
        configuration = LEGACY_CONFIGURATIONS[
            modelIndex % len(LEGACY_CONFIGURATIONS)
        ]
        model = createDeepLearningRegressor(
            randomState=SEED_BASE + FOLD_INDEX * 100 + modelIndex,
            useEarlyStopping=useEarlyStopping,
            configuration=configuration,
        )
        model.fit(trainX, trainY)
        predictions.append(clipPredictionsToTrainingBounds(
            model.predict(validationX), trainY
        ))
    return np.median(predictions, axis=0)


def trainFusionExpert(trainX, trainY, validationX, configurations):
    predictions = []
    for modelIndex in range(TORCH_MODELS):
        configuration = configurations[modelIndex % len(configurations)]
        trainingResult = torchService.trainTorchModel(
            trainX,
            trainY,
            configuration,
            SEED_BASE + 10000 + FOLD_INDEX * 100 + modelIndex,
            TORCH_EPOCHS,
        )
        prediction = torchService.predictTorchModel(
            trainingResult, validationX
        )
        predictions.append(torchService.clipToTrainingBounds(
            prediction, trainY
        ))
    return np.median(predictions, axis=0)


def main():
    if not 0 <= FOLD_INDEX < FOLDS:
        raise ValueError(f'Invalid fold {FOLD_INDEX}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with app.app_context():
        channelFeatures, channelTargets = loadReferenceMatrix(37)
        fusionFeatures, fusionTargets = loadFusionMatrix()
    if not np.allclose(channelTargets, fusionTargets):
        raise ValueError('Channel 37 and fusion samples are not aligned')
    groups = positionGroups(fusionTargets)
    splits = list(GroupKFold(n_splits=FOLDS).split(
        fusionFeatures, fusionTargets, groups
    ))
    trainIndexes, validationIndexes = splits[FOLD_INDEX]
    configurations = loadFusionConfigurations()
    started = time.perf_counter()
    legacyPrediction = trainLegacyExpert(
        channelFeatures[trainIndexes],
        channelTargets[trainIndexes],
        channelFeatures[validationIndexes],
    )
    fusionPrediction = trainFusionExpert(
        fusionFeatures[trainIndexes],
        fusionTargets[trainIndexes],
        fusionFeatures[validationIndexes],
        configurations,
    )
    payload = {
        'fold': FOLD_INDEX,
        'folds': FOLDS,
        'torchModels': TORCH_MODELS,
        'torchEpochs': TORCH_EPOCHS,
        'seconds': round(time.perf_counter() - started, 3),
        'indexes': validationIndexes.tolist(),
        'targets': fusionTargets[validationIndexes].tolist(),
        'legacyPredictions': legacyPrediction.tolist(),
        'fusionPredictions': fusionPrediction.tolist(),
    }
    outputPath = OUTPUT_DIR / f'pytorch_hybrid_fold_{FOLD_INDEX}.json'
    temporaryPath = outputPath.with_suffix('.tmp')
    temporaryPath.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporaryPath.replace(outputPath)
    print(json.dumps({
        'fold': FOLD_INDEX,
        'samples': len(validationIndexes),
        'seconds': payload['seconds'],
    }), flush=True)


if __name__ == '__main__':
    main()
