import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import app
from controllers.CampaignCrud import (
    getPointsByCampaignId,
    getRelatedCampaignIdByCampaignId,
)
from controllers.MethodPredataCrud import getUniqueCoordinatesByCampaignId
from services.DataProcessService import (
    buildEvaluationSamplePlan,
    getAlePointsMatrix,
    getConfiguredRotations,
    getDeepLearningRefPointsMatrix,
    processBeaconMacs,
)
from services.MethodsService import prepareDeepLearningFeatures
import services.PyTorchPositioningService as torchService


CAMPAIGN_ID = int(os.getenv('IPSCM_CAMPAIGN_ID', '2'))
PROTOCOL = os.getenv('IPSCM_PROTOCOL', 'iBeacon')
CHANNELS = [
    int(value)
    for value in os.getenv('IPSCM_CHANNELS', '37,38,39').split(',')
]
RSSI_SAMPLES = int(os.getenv('IPSCM_RSSI_SAMPLES', '20'))
TRIALS = int(os.getenv('IPSCM_SEARCH_TRIALS', '120'))
FOLDS = int(os.getenv('IPSCM_SEARCH_FOLDS', '5'))
EPOCHS = int(os.getenv('IPSCM_SEARCH_EPOCHS', '700'))
OUTPUT_DIR = Path(os.getenv('IPSCM_RESULTS_DIR', '/app/Resultados'))
SEARCH_SEED = 20260701
FUSION = os.getenv('IPSCM_FUSION', '0') == '1'
SEARCH_MODE = os.getenv('IPSCM_SEARCH_MODE', 'broad')


def generateConfiguration(rng):
    return {
        'width': int(rng.choice([32, 64, 96, 128, 192, 256, 384])),
        'blocks': int(rng.choice([1, 2, 3, 4, 5])),
        'dropout': float(rng.uniform(0.0, 0.35)),
        'learningRate': float(
            np.exp(rng.uniform(np.log(1e-4), np.log(8e-3)))
        ),
        'weightDecay': float(
            np.exp(rng.uniform(np.log(1e-7), np.log(3e-2)))
        ),
        'huberDelta': float(rng.uniform(0.2, 2.5)),
        'tailFraction': float(rng.choice([0.1, 0.15, 0.2, 0.3])),
        'tailWeight': float(rng.uniform(0.0, 0.9)),
        'noiseStd': float(rng.choice([0.0, 0.01, 0.02, 0.04, 0.08])),
    }


def generateFocusedConfiguration(rng):
    return {
        'width': int(rng.choice([24, 32, 48, 64, 96])),
        'blocks': int(rng.choice([1, 2, 3, 4, 5, 6])),
        'dropout': float(rng.uniform(0.15, 0.40)),
        'learningRate': float(
            np.exp(rng.uniform(np.log(5e-5), np.log(7e-4)))
        ),
        'weightDecay': float(
            np.exp(rng.uniform(np.log(1e-7), np.log(5e-2)))
        ),
        'huberDelta': float(rng.uniform(0.4, 2.8)),
        'tailFraction': float(rng.choice([0.1, 0.15, 0.2, 0.25, 0.3])),
        'tailWeight': float(rng.uniform(0.15, 0.95)),
        'noiseStd': float(rng.choice([0.02, 0.04, 0.06, 0.08, 0.12])),
    }


def loadPreviousResults(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as inputFile:
        return json.load(inputFile).get('trials', [])


def saveResults(path, channel, trials):
    ordered = sorted(trials, key=lambda item: item['score'])
    payload = {
        'campaignId': CAMPAIGN_ID,
        'protocol': PROTOCOL,
        'channel': channel,
        'rssiSamples': RSSI_SAMPLES,
        'folds': FOLDS,
        'epochs': EPOCHS,
        'best': ordered[0] if ordered else None,
        'trials': ordered,
    }
    temporaryPath = path.with_suffix('.tmp')
    temporaryPath.write_text(
        json.dumps(payload, indent=2),
        encoding='utf-8',
    )
    temporaryPath.replace(path)


def loadReferenceMatrix(channel):
    pointsConfiguration = getPointsByCampaignId(CAMPAIGN_ID)
    referenceConfiguration = pointsConfiguration['Ref_points']
    aleatoryCampaignId = getRelatedCampaignIdByCampaignId(CAMPAIGN_ID)
    referencePoints = getUniqueCoordinatesByCampaignId(
        CAMPAIGN_ID, len(referenceConfiguration)
    )
    beaconMacs = processBeaconMacs(
        CAMPAIGN_ID, aleatoryCampaignId, PROTOCOL
    )
    matrix = getDeepLearningRefPointsMatrix(
        referencePoints,
        referenceConfiguration,
        beaconMacs,
        CAMPAIGN_ID,
        PROTOCOL,
        channel,
        RSSI_SAMPLES,
        set(getConfiguredRotations(referenceConfiguration)),
    )
    return (
        prepareDeepLearningFeatures(matrix[:, :-3]),
        matrix[:, -3:].astype(np.float32),
    )


def loadEvaluationMatrix(channel):
    pointsConfiguration = getPointsByCampaignId(CAMPAIGN_ID)
    aleatoryConfiguration = pointsConfiguration['Ale_points']
    aleatoryCampaignId = getRelatedCampaignIdByCampaignId(CAMPAIGN_ID)
    aleatoryPoints = getUniqueCoordinatesByCampaignId(
        aleatoryCampaignId, len(aleatoryConfiguration)
    )
    beaconMacs = processBeaconMacs(
        CAMPAIGN_ID, aleatoryCampaignId, PROTOCOL
    )
    evaluationSamplePlan = buildEvaluationSamplePlan(
        aleatoryPoints, aleatoryConfiguration, beaconMacs,
        aleatoryCampaignId, PROTOCOL
    )
    matrix = getAlePointsMatrix(
        aleatoryPoints,
        aleatoryConfiguration,
        beaconMacs,
        aleatoryCampaignId,
        PROTOCOL,
        channel,
        RSSI_SAMPLES,
        evaluationSamplePlan,
    )
    return (
        prepareDeepLearningFeatures(matrix[:, :-3]),
        matrix[:, -3:].astype(np.float32),
    )


def positionGroups(targets):
    _, groups = np.unique(
        np.round(np.asarray(targets, dtype=float), 6),
        axis=0,
        return_inverse=True,
    )
    return groups


def runSearch(label, features, targets, seedOffset):
    suffix = '' if SEARCH_MODE == 'broad' else f'_{SEARCH_MODE}'
    outputPath = OUTPUT_DIR / f'pytorch_search_{label}{suffix}.json'
    trials = loadPreviousResults(outputPath)
    groups = positionGroups(targets)
    splitter = GroupKFold(
        n_splits=min(FOLDS, len(np.unique(groups)))
    )
    rng = np.random.default_rng(SEARCH_SEED + seedOffset)
    baseline = {
        'width': 128,
        'blocks': 2,
        'dropout': 0.1,
        'learningRate': 0.003,
        'weightDecay': 0.001,
        'huberDelta': 0.75,
        'tailFraction': 0.2,
        'tailWeight': 0.35,
        'noiseStd': 0.0,
    }
    if SEARCH_MODE == 'focused':
        broadPath = OUTPUT_DIR / f'pytorch_search_{label}.json'
        broadResults = json.loads(
            broadPath.read_text(encoding='utf-8')
        )
        baseline = broadResults['best']['configuration']
        generator = generateFocusedConfiguration
    else:
        generator = generateConfiguration
    configurations = [baseline] + [
        generator(rng) for _ in range(max(0, TRIALS - 1))
    ]
    print(
        f'dataset={label} samples={len(features)} '
        f'features={features.shape[1]} mode={SEARCH_MODE} '
        f'completed={len(trials)}',
        flush=True,
    )
    for trialIndex in range(len(trials), TRIALS):
        configuration = configurations[trialIndex]
        started = time.perf_counter()
        score = torchService.scoreTorchConfiguration(
            configuration,
            features,
            targets,
            groups,
            splitter,
            trialIndex,
            maxEpochs=EPOCHS,
        )
        elapsed = time.perf_counter() - started
        trial = {
            'index': trialIndex,
            'score': float(score),
            'seconds': round(elapsed, 3),
            'configuration': configuration,
        }
        trials.append(trial)
        saveResults(outputPath, label, trials)
        print(json.dumps(trial), flush=True)


def loadFusionMatrix():
    channelMatrices = [
        loadReferenceMatrix(channel) for channel in CHANNELS
    ]
    targets = channelMatrices[0][1]
    for _, channelTargets in channelMatrices[1:]:
        if not np.allclose(targets, channelTargets):
            raise ValueError(
                'Channel fingerprints are not spatially aligned'
            )
    features = np.concatenate(
        [features for features, _ in channelMatrices],
        axis=1,
    )
    return features, targets


def loadFusionEvaluationMatrix():
    channelMatrices = [
        loadEvaluationMatrix(channel) for channel in CHANNELS
    ]
    targets = channelMatrices[0][1]
    for _, channelTargets in channelMatrices[1:]:
        if not np.allclose(targets, channelTargets):
            raise ValueError(
                'Evaluation points are not aligned between channels'
            )
    features = np.concatenate(
        [features for features, _ in channelMatrices],
        axis=1,
    )
    return features, targets


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f'device={torchService.getTorchDevice()} '
        f'trials={TRIALS} folds={FOLDS} epochs={EPOCHS}',
        flush=True,
    )
    with app.app_context():
        if FUSION:
            features, targets = loadFusionMatrix()
            runSearch('fusion_37_38_39', features, targets, 373839)
        else:
            for channel in CHANNELS:
                features, targets = loadReferenceMatrix(channel)
                runSearch(
                    f'channel_{channel}',
                    features,
                    targets,
                    channel,
                )


if __name__ == '__main__':
    main()
