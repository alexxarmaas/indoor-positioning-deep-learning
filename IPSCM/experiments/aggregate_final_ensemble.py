import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.optimize_pytorch import OUTPUT_DIR


SHARD_COUNT = 4
EXPECTED_MODELS = 48
BOOTSTRAP_SAMPLES = 10000


def errorSummary(targets, predictions):
    errors = np.linalg.norm(targets - predictions, axis=1)
    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'p90': float(np.percentile(errors, 90)),
        'maximum': float(np.max(errors)),
        'errors': errors.tolist(),
    }


def bootstrapMeanInterval(errors):
    rng = np.random.default_rng(20260702)
    indexes = rng.integers(
        0, len(errors), size=(BOOTSTRAP_SAMPLES, len(errors))
    )
    means = np.mean(errors[indexes], axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def main():
    shards = []
    for shardIndex in range(SHARD_COUNT):
        path = OUTPUT_DIR / (
            f'pytorch_final_fusion_shard_{shardIndex}.json'
        )
        shards.append(json.loads(path.read_text(encoding='utf-8')))
    targets = np.asarray(shards[0]['targets'], dtype=float)
    items = [
        item
        for shard in shards
        for item in shard['predictions']
    ]
    items.sort(key=lambda item: item['modelIndex'])
    if len(items) != EXPECTED_MODELS:
        raise ValueError(
            f'Expected {EXPECTED_MODELS} models, found {len(items)}'
        )
    predictions = np.asarray(
        [item['values'] for item in items], dtype=float
    )
    medianPrediction = np.median(predictions, axis=0)
    meanPrediction = np.mean(predictions, axis=0)
    medianSummary = errorSummary(targets, medianPrediction)
    meanSummary = errorSummary(targets, meanPrediction)
    medianSummary['mean95CI'] = bootstrapMeanInterval(
        np.asarray(medianSummary['errors'])
    )
    payload = {
        'models': len(items),
        'primaryAggregation': 'coordinate-wise median',
        'medianEnsemble': medianSummary,
        'meanEnsemble': meanSummary,
        'targets': targets.tolist(),
        'predictions': medianPrediction.tolist(),
    }
    outputPath = OUTPUT_DIR / 'pytorch_final_fusion_metrics.json'
    outputPath.write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
