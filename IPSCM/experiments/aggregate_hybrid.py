import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.optimize_pytorch import OUTPUT_DIR


FOLDS = 5
WEIGHTS = np.linspace(0.0, 1.0, 21)


def summarize(targets, predictions):
    errors = np.linalg.norm(targets - predictions, axis=1)
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'maximum': float(np.max(errors)),
        'objective': float(
            np.mean(errors) + 0.5 * np.percentile(errors, 90)
        ),
    }


def main():
    folds = [
        json.loads((OUTPUT_DIR / f'pytorch_hybrid_fold_{index}.json')
                   .read_text(encoding='utf-8'))
        for index in range(FOLDS)
    ]
    targets = np.concatenate([
        np.asarray(item['targets'], dtype=float) for item in folds
    ])
    legacy = np.concatenate([
        np.asarray(item['legacyPredictions'], dtype=float) for item in folds
    ])
    fusion = np.concatenate([
        np.asarray(item['fusionPredictions'], dtype=float) for item in folds
    ])
    trials = []
    for fusionWeight in WEIGHTS:
        prediction = (
            fusionWeight * fusion + (1.0 - fusionWeight) * legacy
        )
        trial = summarize(targets, prediction)
        trial['fusionWeight'] = float(fusionWeight)
        trial['legacyWeight'] = float(1.0 - fusionWeight)
        trials.append(trial)
    trials.sort(key=lambda item: item['objective'])
    payload = {
        'selectionData': 'reference fingerprints only; grouped 5-fold OOF',
        'best': trials[0],
        'legacyOnly': next(
            item for item in trials if item['fusionWeight'] == 0.0
        ),
        'fusionOnly': next(
            item for item in trials if item['fusionWeight'] == 1.0
        ),
        'trials': trials,
    }
    outputPath = OUTPUT_DIR / 'pytorch_hybrid_validation.json'
    outputPath.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
