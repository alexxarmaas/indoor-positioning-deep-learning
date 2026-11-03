import sqlite3
import numpy as np 
from pathlib import Path
from types import SimpleNamespace

from services.TFGFeaturesService import build_tfg_dataset

DB_PATH = Path("/tmp/toma_datos_16_4.sqlite3")

with sqlite3.connect(DB_PATH) as conn:

    captures = [
        SimpleNamespace(
            Id=row[0],
            Position_x=row[1],
            Position_y=row[2]
        )
        for row in conn.execute("""
        SELECT id, Position_x, Position_y
        FROM Capture
        ORDER BY Id
        """
        )
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

print("Capturas originales: ", len(captures))
print("Fingerprint generados: ", len(X))
print("Dimensiones X: ", X.shape)
print("Dimensiones y: ", y.shape)
print("Identificadores: ", capture_ids.shape)

empty_fingerprints = np.sum(
    np.sum(X[:, 8:16], axis = 1) == 0
)

print("Fingerprints sin balizas: ", empty_fingerprints)

assert X.shape == (239, 16)
assert y.shape == (239, 2)
assert len(capture_ids) == 239
assert empty_fingerprints == 0

print("Validación completada correctamente")