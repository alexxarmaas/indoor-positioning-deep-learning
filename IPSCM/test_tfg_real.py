import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from services.TFGFeaturesService import build_tfg_dataset
from services.TFGPositioningService import (build_tfg_mlp, apply_tfg_mlp_method)
from services.MethodsService import applyMethod

DB_PATH = Path("/tmp/toma_datos_16_4.sqlite3")

with sqlite3.connect(DB_PATH) as conn:

    points = conn.execute("""
        SELECT Position_x, Position_y, Position_z
        FROM Capture
        GROUP BY Position_x, Position_y, Position_z
        ORDER BY MIN(Id)
    """
    ).fetchall()

    captures = [
        SimpleNamespace(
            Id = row[0],
            Position_x = row[1],
            Position_y = row[2],
            Position_z = row[3]
        )
        for row in conn.execute("""
            SELECT Id, Position_x, Position_y, Position_z
            FROM Capture
            ORDER BY Id
        """)
    ]

    signals = [
        SimpleNamespace(
            Id_capture = row[0],
            Mac = row[1],
            RSSI = row[2]
        )
        for row in conn.execute("""
            SELECT Id_capture, Mac, RSSI
            FROM Beacon_BLE_Signal
        """
        )
    ]

X, y, capture_ids = build_tfg_dataset(captures, signals)

print("Fingerprints: ", X.shape)
print("Coordenadas: ", y.shape)

point_mapping = {
    tuple(point): index
    for index, point in enumerate(points, start=1)
}

capture_mapping = {
    capture.Id: (
        capture.Position_x,
        capture.Position_y,
        capture.Position_z
    )
    for capture in captures
}

point_numbers = np.asarray([
    point_mapping[capture_mapping[int(capture_id)]]
    for capture_id in capture_ids
])

train_mask = point_numbers % 2 == 1
test_mask = ~train_mask

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

test_points = point_numbers[test_mask]

assert len(points) == 60
assert len(X_train) == 119
assert len(X_test) == 120

assert X_train.shape == (119, 16)
assert X_test.shape == (120, 16)

assert y_train.shape == (119, 2)
assert y_test.shape == (120, 2)

assert np.isfinite(X_train).all()
assert np.isfinite(X_test).all()
assert np.isfinite(y_train).all()
assert np.isfinite(y_test).all()

assert set(point_numbers[train_mask]).isdisjoint(set(point_numbers[test_mask]))

print("Entrenamiento: ", X_train.shape)
print("Evaluación: ", X_test.shape)

z_values = np.asarray([
    capture_mapping[int(capture_id)][2]
    for capture_id in capture_ids
])

data_train = np.column_stack((
    X_train, y_train, z_values[train_mask]
))

data_test = np.column_stack((
    X_test, y_test, z_values[test_mask]
))

print("--- Prueba IPSCampaign ---")
print("data_train: ", data_train.shape)
print("data_test: ", data_test.shape)

for use_mask in [False, True]:

    method_name = ("MLP_TFG_MASK" if use_mask else "MLP_TFG_ROBUST")
    results = applyMethod(
        data_train=data_train, 
        data_test=data_test,
        method=method_name,
        metrics=None, 
        algorithms=None,
        ks_range=None,
        kernel=None,
        cs=None, 
        gammas=None,
        nus=None,
        Is=None, 
        description="{}")
        
    predictions_xyz = np.asarray(results[method_name], dtype=float)

    assert predictions_xyz.shape == (len(X_test), 3)
    assert np.isfinite(predictions_xyz).all()

    assert np.allclose(predictions_xyz[:, 2], z_values[train_mask][0])
    print(f"{method_name}: {predictions_xyz.shape} OK")


def calculate_metrics(y_true, y_pred):

    errors = np.linalg.norm(y_true - y_pred, axis = 1)

    return {
        "Error medio": float(np.mean(errors)),
        "Mediana": float(np.median(errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "P90": float(np.percentile(errors, 90)),
        "Máximo": float(np.max(errors))
    }

for use_mask in [False, True]:

    model_name = (
        "MLP_TFG_MASK" if use_mask else "MLP_TFG_ROBUST"
    )

    print(f"---{model_name}---")

    model = build_tfg_mlp(use_mask=use_mask)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    adapted_results = apply_tfg_mlp_method(data_train, data_test, "{}", use_mask=use_mask)
    adapted_xy = np.asarray(adapted_results[model_name])[:, :2]
    np.testing.assert_allclose(predictions, adapted_xy, rtol=1e-7, atol=1e-7)

    capture_metrics = calculate_metrics(y_test, predictions)

    fused_real = []
    fused_predicted = []

    for point in np.unique(test_points):

        mask = test_points == point
        real_coordinates = y_test[mask]

        if not np.allclose(real_coordinates, real_coordinates[0]):
            raise ValueError(f"Coordenadas inconsistentes en el punto {point}")

        fused_real.append(real_coordinates[0])
        fused_predicted.append(np.median(predictions[mask], axis=0))

    fused_metrics = calculate_metrics(np.asarray(fused_real), np.asarray(fused_predicted))

    print("Resultados por captura: ")
    for metric, value in capture_metrics.items():
        print(f" {metric}: {value:.4f} m")

    print("Resultados fusionados por punto: ")
    for metric, value in fused_metrics.items():
        print(f" {metric}: {value:.4f} m")
