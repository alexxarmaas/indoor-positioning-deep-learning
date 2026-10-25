from TFGFeaturesService import build_tfg_fingerprint
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MLP_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 2000,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 50,
    "random_state": 42
}

def build_tfg_mlp(use_mask=False):

    transformers = [("rssi", StandardScaler(), list(range(8)))]

    if use_mask:
        transformers.append(("presence", "passthrough", list(range(8, 16))))
    
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = MLPRegressor(**MLP_CONFIG)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    
    return pipeline

