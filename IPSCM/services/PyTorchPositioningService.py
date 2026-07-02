import copy
import math

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn


TORCH_MODEL_CANDIDATES = (
    {'width': 64, 'blocks': 2, 'dropout': 0.05},
    {'width': 128, 'blocks': 2, 'dropout': 0.10},
    {'width': 128, 'blocks': 3, 'dropout': 0.15},
)
TORCH_ENSEMBLE_SIZE = 6
TORCH_TOP_CONFIGURATIONS = 2
TORCH_SELECTION_FOLDS = 3
TORCH_SELECTION_MIN_SAMPLES = 48
TORCH_SELECTION_MIN_GROUPS = 6
TORCH_MAX_EPOCHS = 1200
TORCH_CV_EPOCHS = 500
TORCH_PATIENCE = 120
TORCH_LEARNING_RATE = 0.003
TORCH_WEIGHT_DECAY = 0.001
TORCH_HUBER_DELTA_METERS = 0.75
TORCH_TAIL_FRACTION = 0.20
TORCH_TAIL_WEIGHT = 0.35
TORCH_P90_WEIGHT = 0.5
TORCH_RANDOM_STATE = 45


def getTorchDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.SiLU()

    def forward(self, values):
        return self.activation(values + self.network(values))


class PositioningNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, configuration):
        super().__init__()
        width = configuration['width']
        self.inputLayer = nn.Sequential(
            nn.Linear(inputSize, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(width, configuration['dropout'])
            for _ in range(configuration['blocks'])
        ])
        self.outputLayer = nn.Linear(width, outputSize)

    def forward(self, values):
        return self.outputLayer(self.blocks(self.inputLayer(values)))


def setTorchSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def robustPositioningLoss(
        prediction, target, targetScale, configuration):
    physicalDifference = (prediction - target) * targetScale
    distances = torch.linalg.vector_norm(physicalDifference, dim=1)
    delta = configuration.get(
        'huberDelta', TORCH_HUBER_DELTA_METERS
    )
    robustError = torch.where(
        distances <= delta,
        0.5 * distances.square() / delta,
        distances - 0.5 * delta,
    )
    tailFraction = configuration.get(
        'tailFraction', TORCH_TAIL_FRACTION
    )
    tailWeight = configuration.get('tailWeight', TORCH_TAIL_WEIGHT)
    tailCount = max(1, math.ceil(len(distances) * tailFraction))
    tailError = torch.topk(distances, tailCount).values.mean()
    return robustError.mean() + tailWeight * tailError


def standardizeValues(values):
    values = np.asarray(values, dtype=np.float32)
    mean = np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (values - mean) / scale, mean, scale


def toTorchTensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def trainTorchModel(features, targets, configuration, seed, maxEpochs):
    setTorchSeed(seed)
    device = getTorchDevice()
    normalizedX, xMean, xScale = standardizeValues(features)
    normalizedY, yMean, yScale = standardizeValues(targets)
    xTensor = toTorchTensor(normalizedX, device)
    yTensor = toTorchTensor(normalizedY, device)
    yScaleTensor = toTorchTensor(yScale, device)
    model = PositioningNetwork(
        normalizedX.shape[1], normalizedY.shape[1], configuration
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.get('learningRate', TORCH_LEARNING_RATE),
        weight_decay=configuration.get(
            'weightDecay', TORCH_WEIGHT_DECAY
        ),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=maxEpochs,
        eta_min=configuration.get(
            'learningRate', TORCH_LEARNING_RATE
        ) * 0.05,
    )
    bestLoss = float('inf')
    bestState = copy.deepcopy(model.state_dict())
    staleEpochs = 0

    for _ in range(maxEpochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noiseStd = configuration.get('noiseStd', 0.0)
        modelInput = xTensor
        if noiseStd > 0:
            modelInput = xTensor + torch.randn_like(xTensor) * noiseStd
        prediction = model(modelInput)
        loss = robustPositioningLoss(
            prediction, yTensor, yScaleTensor, configuration
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        currentLoss = float(loss.detach().cpu())
        if currentLoss < bestLoss - 1e-5:
            bestLoss = currentLoss
            bestState = copy.deepcopy(model.state_dict())
            staleEpochs = 0
        else:
            staleEpochs += 1
        if staleEpochs >= TORCH_PATIENCE:
            break

    model.load_state_dict(bestState)
    model.eval()
    return model, xMean, xScale, yMean, yScale, device


def predictTorchModel(trainingResult, features):
    model, xMean, xScale, yMean, yScale, device = trainingResult
    normalizedX = (np.asarray(features, dtype=np.float32) - xMean) / xScale
    with torch.inference_mode():
        prediction = model(toTorchTensor(normalizedX, device)).cpu().numpy()
    return prediction * yScale + yMean


def clipToTrainingBounds(predictions, targets):
    return np.clip(
        predictions,
        np.min(targets, axis=0),
        np.max(targets, axis=0),
    )


def scoreTorchConfiguration(
        configuration, features, targets, groups, splitter, candidateIndex,
        maxEpochs=None):
    maxEpochs = maxEpochs or TORCH_CV_EPOCHS
    foldScores = []
    for trainIndexes, validationIndexes in splitter.split(
            features, targets, groups):
        trainingResult = trainTorchModel(
            features[trainIndexes],
            targets[trainIndexes],
            configuration,
            TORCH_RANDOM_STATE + candidateIndex,
            maxEpochs,
        )
        prediction = predictTorchModel(
            trainingResult, features[validationIndexes]
        )
        prediction = clipToTrainingBounds(
            prediction, targets[trainIndexes]
        )
        error = np.linalg.norm(
            targets[validationIndexes] - prediction, axis=1
        )
        foldScores.append(
            np.mean(error) + TORCH_P90_WEIGHT * np.percentile(error, 90)
        )
    return np.mean(foldScores)


def selectTorchConfigurations(features, targets):
    _, groups = np.unique(
        np.round(np.asarray(targets, dtype=float), 6),
        axis=0,
        return_inverse=True,
    )
    groupCount = len(np.unique(groups))
    if (
        len(features) < TORCH_SELECTION_MIN_SAMPLES
        or groupCount < TORCH_SELECTION_MIN_GROUPS
    ):
        return [TORCH_MODEL_CANDIDATES[0]]
    splitter = GroupKFold(
        n_splits=min(TORCH_SELECTION_FOLDS, groupCount)
    )
    scores = [
        (
            scoreTorchConfiguration(
                candidate, features, targets, groups, splitter, index
            ),
            candidate,
        )
        for index, candidate in enumerate(TORCH_MODEL_CANDIDATES)
    ]
    scores.sort(key=lambda item: item[0])
    return [
        candidate for _, candidate in scores[:TORCH_TOP_CONFIGURATIONS]
    ]


def predictWithPyTorchEnsemble(features, targets, testFeatures):
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    testFeatures = np.asarray(testFeatures, dtype=np.float32)
    configurations = selectTorchConfigurations(features, targets)
    predictions = []
    for ensembleIndex in range(TORCH_ENSEMBLE_SIZE):
        configuration = configurations[
            ensembleIndex % len(configurations)
        ]
        trainingResult = trainTorchModel(
            features,
            targets,
            configuration,
            TORCH_RANDOM_STATE + ensembleIndex,
            TORCH_MAX_EPOCHS,
        )
        prediction = predictTorchModel(trainingResult, testFeatures)
        predictions.append(clipToTrainingBounds(prediction, targets))
    return np.median(predictions, axis=0), configurations


def getPyTorchConfiguration():
    return {
        'framework': 'PyTorch',
        'architecture': 'ResidualMLP',
        'candidateModels': list(TORCH_MODEL_CANDIDATES),
        'selectionFolds': TORCH_SELECTION_FOLDS,
        'selectionObjective': f'mean + {TORCH_P90_WEIGHT} * P90',
        'ensembleSize': TORCH_ENSEMBLE_SIZE,
        'ensembleAggregation': 'coordinate-wise median',
        'optimizer': 'AdamW',
        'maxEpochs': TORCH_MAX_EPOCHS,
        'learningRate': TORCH_LEARNING_RATE,
        'weightDecay': TORCH_WEIGHT_DECAY,
        'loss': {
            'type': 'euclidean Huber + tail penalty',
            'huberDeltaMeters': TORCH_HUBER_DELTA_METERS,
            'tailFraction': TORCH_TAIL_FRACTION,
            'tailWeight': TORCH_TAIL_WEIGHT,
        },
        'device': str(getTorchDevice()),
        'randomState': TORCH_RANDOM_STATE,
    }
