from services.TFGFeaturesService import build_tfg_fingerprint
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np


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

def apply_tfg_mlp_method(data_train, data_test, description, use_mask=False):

    data_train = np.asarray(data_train, dtype=float)
    data_test = np.asarray(data_test, dtype=float)

    for name, matrix in [("data_train", data_train), ("data_test", data_test)]:
        if matrix.ndim != 2 or matrix.shape[1] != 19:
            raise ValueError(f"{name} debe tener 19 columnas")

        if len(matrix) == 0:
            raise ValueError(f"{name} no puede estar vacía")
        
    X_train = data_train[:, :16]
    y_train = data_train[:, 16:18]

    X_test = data_test[:, :16]

    if not np.isfinite(X_train).all():
        raise ValueError("Características de entrenamiento no válidas")

    if not np.isfinite(y_train).all():
        raise ValueError("Coordenadas de entrenamiento no válidas")

    if not np.isfinite(X_test).all():
        raise ValueError("Características de evaluación no válidas")

    z_train = data_train[:, 18]

    if not np.isfinite(z_train).all():
        raise ValueError("Alturas de entrenamiento no válidas")

    if not np.allclose(z_train, z_train[0], atol=1e-4, rtol=0):
        raise ValueError("El modelo TFG requiere una altura Z constante")

    model = build_tfg_mlp(use_mask=use_mask)
    model.fit(X_train, y_train)

    predictions_xy = model.predict(X_test)

    if predictions_xy.shape != (len(X_test), 2):
        raise ValueError("Dimensiones de predicción incorrectas")

    if not np.isfinite(predictions_xy).all():
        raise ValueError("Las predicciones contienen valores no finitos")

    z = float(z_train[0])

    predictions_xyz = np.column_stack((predictions_xy, np.full(len(predictions_xy), z)))

    method_name = ("MLP_TFG_MASK" if use_mask else "MLP_TFG_ROBUST")

    return {method_name: predictions_xyz.tolist()}