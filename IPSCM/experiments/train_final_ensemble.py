import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import app
from experiments.optimize_pytorch import (
    OUTPUT_DIR,
    loadFusionEvaluationMatrix,
    loadFusionMatrix,
)
import services.PyTorchPositioningService as torchService


TOTAL_MODELS = int(os.getenv('IPSCM_FINAL_MODELS', '48'))
SHARD_COUNT = int(os.getenv('IPSCM_FINAL_SHARDS', '4'))
SHARD_INDEX = int(os.getenv('IPSCM_FINAL_SHARD', '0'))
EPOCHS = int(os.getenv('IPSCM_FINAL_EPOCHS', '1800'))
TOP_CONFIGURATIONS = int(os.getenv('IPSCM_FINAL_TOP', '3'))
SEED_BASE = 50000


def loadConfirmedConfigurations():
    path = OUTPUT_DIR / (
        'pytorch_confirmation_fusion_37_38_39.json'
    )
    confirmation = json.loads(path.read_text(encoding='utf-8'))
    summaries = [
        summary for summary in confirmation['summaries']
        if summary['completedRepeats'] == confirmation['repeats']
    ]
    summaries.sort(
        key=lambda item: item['meanScore'] + item['stdScore']
    )
    return [
        summary['configuration']
        for summary in summaries[:TOP_CONFIGURATIONS]
    ]


def saveShard(path, predictions, targets, configurations):
    payload = {
        'shardIndex': SHARD_INDEX,
        'shardCount': SHARD_COUNT,
        'totalModels': TOTAL_MODELS,
        'epochs': EPOCHS,
        'configurations': configurations,
        'targets': targets.tolist(),
        'predictions': predictions,
    }
    temporaryPath = path.with_suffix('.tmp')
    temporaryPath.write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )
    temporaryPath.replace(path)


def main():
    outputPath = OUTPUT_DIR / (
        f'pytorch_final_fusion_shard_{SHARD_INDEX}.json'
    )
    configurations = loadConfirmedConfigurations()
    with app.app_context():
        trainFeatures, trainTargets = loadFusionMatrix()
        testFeatures, testTargets = loadFusionEvaluationMatrix()
    predictions = []
    if outputPath.exists():
        predictions = json.loads(
            outputPath.read_text(encoding='utf-8')
        ).get('predictions', [])
    completedIndexes = {
        item['modelIndex'] for item in predictions
    }
    modelIndexes = range(SHARD_INDEX, TOTAL_MODELS, SHARD_COUNT)
    print(
        f'device={torchService.getTorchDevice()} shard={SHARD_INDEX} '
        f'models={len(list(modelIndexes))} epochs={EPOCHS}',
        flush=True,
    )
    for modelIndex in modelIndexes:
        if modelIndex in completedIndexes:
            continue
        configurationIndex = modelIndex % len(configurations)
        configuration = configurations[configurationIndex]
        started = time.perf_counter()
        trainingResult = torchService.trainTorchModel(
            trainFeatures,
            trainTargets,
            configuration,
            SEED_BASE + modelIndex,
            EPOCHS,
        )
        prediction = torchService.predictTorchModel(
            trainingResult, testFeatures
        )
        prediction = torchService.clipToTrainingBounds(
            prediction, trainTargets
        )
        item = {
            'modelIndex': modelIndex,
            'configurationIndex': configurationIndex,
            'seconds': round(time.perf_counter() - started, 3),
            'values': prediction.tolist(),
        }
        predictions.append(item)
        saveShard(
            outputPath, predictions, testTargets, configurations
        )
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
