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
    CHANNELS,
    FUSION,
    OUTPUT_DIR,
    loadFusionMatrix,
    loadReferenceMatrix,
    positionGroups,
)
import services.PyTorchPositioningService as torchService


TOP_CONFIGURATIONS = int(os.getenv('IPSCM_CONFIRM_TOP', '5'))
REPEATS = int(os.getenv('IPSCM_CONFIRM_REPEATS', '5'))
FOLDS = int(os.getenv('IPSCM_CONFIRM_FOLDS', '5'))
EPOCHS = int(os.getenv('IPSCM_CONFIRM_EPOCHS', '1200'))


def loadCandidates(label):
    sourcePath = OUTPUT_DIR / (
        f'pytorch_search_{label}_focused.json'
    )
    searchResults = json.loads(
        sourcePath.read_text(encoding='utf-8')
    )
    ordered = sorted(
        searchResults['trials'], key=lambda item: item['score']
    )
    return [
        trial['configuration']
        for trial in ordered[:TOP_CONFIGURATIONS]
    ]


def saveConfirmation(path, label, candidates, runs):
    summaries = []
    for candidateIndex, configuration in enumerate(candidates):
        scores = [
            run['score']
            for run in runs
            if run['candidateIndex'] == candidateIndex
        ]
        summaries.append({
            'candidateIndex': candidateIndex,
            'configuration': configuration,
            'completedRepeats': len(scores),
            'meanScore': float(np.mean(scores)) if scores else None,
            'stdScore': float(np.std(scores)) if scores else None,
            'worstScore': float(np.max(scores)) if scores else None,
        })
    completed = [
        item for item in summaries
        if item['completedRepeats'] == REPEATS
    ]
    completed.sort(
        key=lambda item: item['meanScore'] + item['stdScore']
    )
    payload = {
        'dataset': label,
        'repeats': REPEATS,
        'folds': FOLDS,
        'epochs': EPOCHS,
        'best': completed[0] if completed else None,
        'summaries': summaries,
        'runs': runs,
    }
    temporaryPath = path.with_suffix('.tmp')
    temporaryPath.write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )
    temporaryPath.replace(path)


def runConfirmation(label, features, targets):
    outputPath = OUTPUT_DIR / f'pytorch_confirmation_{label}.json'
    candidates = loadCandidates(label)
    if outputPath.exists():
        runs = json.loads(
            outputPath.read_text(encoding='utf-8')
        ).get('runs', [])
    else:
        runs = []
    groups = positionGroups(targets)
    splitter = GroupKFold(
        n_splits=min(FOLDS, len(np.unique(groups)))
    )
    completedKeys = {
        (run['candidateIndex'], run['repeat'])
        for run in runs
    }
    for candidateIndex, configuration in enumerate(candidates):
        for repeat in range(REPEATS):
            if (candidateIndex, repeat) in completedKeys:
                continue
            started = time.perf_counter()
            score = torchService.scoreTorchConfiguration(
                configuration,
                features,
                targets,
                groups,
                splitter,
                10000 + candidateIndex * 100 + repeat,
                maxEpochs=EPOCHS,
            )
            run = {
                'candidateIndex': candidateIndex,
                'repeat': repeat,
                'score': float(score),
                'seconds': round(time.perf_counter() - started, 3),
            }
            runs.append(run)
            saveConfirmation(
                outputPath, label, candidates, runs
            )
            print(json.dumps(run), flush=True)


def main():
    print(
        f'device={torchService.getTorchDevice()} top={TOP_CONFIGURATIONS} '
        f'repeats={REPEATS} folds={FOLDS} epochs={EPOCHS}',
        flush=True,
    )
    with app.app_context():
        if FUSION:
            features, targets = loadFusionMatrix()
            runConfirmation('fusion_37_38_39', features, targets)
        else:
            for channel in CHANNELS:
                features, targets = loadReferenceMatrix(channel)
                runConfirmation(
                    f'channel_{channel}', features, targets
                )


if __name__ == '__main__':
    main()
