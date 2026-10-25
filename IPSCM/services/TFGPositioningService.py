"""MLP de tfg-ips aplicado a matrices de captura IPSCampaign (19 columnas)."""
import json

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TFG_METHODS = {"MLP_TFG_ROBUST", "MLP_TFG_MASK", "WKNN_TFG"}
TFG_MLP_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 2000,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 50,
    "random_state": 42,
}


def make_tfg_mlp(use_mask=False):
    selectors = [("rssi", StandardScaler(), list(range(8)))]
    if use_mask:
        selectors.append(("presence", "passthrough", list(range(8, 16))))
    preprocessor = ColumnTransformer(selectors, remainder="drop")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", MLPRegressor(**TFG_MLP_CONFIG)),
    ])


def validate_tfg_matrix(data, name):
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 19:
        raise ValueError(f"{name}: se esperaban 19 columnas (8 RSSI, 8 máscaras, xyz)")
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: matriz vacía o contiene valores no finitos")
    masks = matrix[:, 8:16]
    if not np.isin(masks, [0.0, 1.0]).all():
        raise ValueError(f"{name}: máscaras de presencia no binarias")
    return matrix


def applyTFGMLPmethod(data_train, data_test, description, use_mask=False):
    train = validate_tfg_matrix(data_train, "Referencia")
    test = validate_tfg_matrix(data_test, "Evaluación")
    if len(train) < 12:
        raise ValueError("El MLP TFG necesita al menos 12 capturas de referencia")

    train_z = train[:, -1]
    test_z = test[:, -1]
    if not np.allclose(train_z, train_z[0], atol=1e-4, rtol=0):
        raise ValueError("El MLP TFG es 2D: el Z de referencia debe ser constante")
    if not np.allclose(test_z, train_z[0], atol=1e-4, rtol=0):
        raise ValueError("El MLP TFG es 2D: Z de referencia y evaluación difieren")

    model = make_tfg_mlp(use_mask=use_mask)
    model.fit(train[:, :16], train[:, -3:-1])
    predicted_xy = np.asarray(model.predict(test[:, :16]), dtype=float)
    if predicted_xy.shape != (len(test), 2) or not np.isfinite(predicted_xy).all():
        raise ValueError("El MLP TFG devolvió coordenadas XY inválidas")
    predicted_xyz = np.column_stack((
        predicted_xy, np.full(len(test), train_z[0], dtype=float)
    ))
    method = json.loads(description).get("method", "MLP_TFG_MASK" if use_mask else "MLP_TFG_ROBUST")
    label = f"[{method}][layers=64-32][seed=42]"
    return {label: predicted_xyz.tolist()}


def tfg_wknn_matrix(matrix):
    """El wKNN original solo admite RSSI+xyz, no las ocho máscaras adicionales."""
    matrix = validate_tfg_matrix(matrix, "WKNN TFG")
    return np.column_stack((matrix[:, :8], matrix[:, -3:]))
