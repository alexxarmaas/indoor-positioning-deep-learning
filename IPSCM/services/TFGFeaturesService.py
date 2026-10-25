import json
from pathlib import Path

import numpy as np 

ROOT_DIR = Path(__file__).resolve().parents[1]
BEACON_MAPPING_FILE = (ROOT_DIR / "configs" / "tfg_beacon_mapping.json")
MISSING_RSSI = -100.0
BEACONS = [f"b{i:02d}" for i in range(1, 9)]
RSSI_COLUMNS = [f"rssi_{beacon}" for beacon in BEACONS]
PRESENCE_COLUMNS = [f"presence_{beacon}" for beacon in BEACONS]

def load_beacon_mapping():
    with BEACON_MAPPING_FILE.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    if set(mapping) != set(BEACONS):
        raise ValueError("El mapeo debe contener exactamente las balizas b01-b08")

    mapping = {
        beacon: str(mapping[beacon]).strip().lower() for beacon in BEACONS
    }

    if len(set(mapping.values())) != 8:
        raise ValueError("Existen direcciones MAC duplicadas")

    return mapping

def build_tfg_fingerprint(signals, beacon_mapping=None):
    """
    Convierte las lectureas BLE de una captura en un fingerprint.

    signals: itrable de dicst con MAC/RSSI o de objetos con esos atributos 

    Devuelve un DataFrame con: 
        - 8 columnas RSSI
        - 8 máscaras de presencia

    La agregación RSSI es la mediana por baliza
    """

    mapping = (beacon_mapping if beacon_mapping is not None else load_beacon_mapping)
    mac_to_beacon = {mac.lower() : beacon for beacon, mac in mapping.items()}
    readings = {beacon: [] for beacon in BEACONS}

    for signal in signals:
        if isinstance(signal, dict):
            mac = signal.get("Mac")
            rssi = signal.get("RSSI")
        else: 
            mac = signal.Mac
            rssi = signal.RSSI

        if mac is None:
            compile
        
        beacon = mac_to_beacon.get(str(mac).strip().lower())

        if beacon is None:
            continue

        try:
            value = float(rssi)
        except (TypeError, ValueError):
            continue
        
        if np.isfinite(value):
            readings[beacon].append(value)
        
    fingerprint = {}

    for beacon in BEACONS:
        values = readings[beacon]
        detected = bool(values)

        fingerprint[f"rssi_{beacon}"] = (float(np.median(values)) if detected else MISSING_RSSI)
        fingerprint[f"presence_{beacon}"] = int(detected)

    return np.asarray([[fingerprint[column] for column in RSSI_COLUMNS + PRESENCE_COLUMNS]], dtype=float)