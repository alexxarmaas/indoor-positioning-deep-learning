"""Fingerprint BLE de TFG: una fila por Capture, mediana por MAC, todos los canales.

El orden de ocho balizas y el suelo -100 dBm deben coincidir con tfg-ips
scripts/02_build_dataset.py y scripts/03_preprocess_dataset.py.
No modifica el procesamiento existente de IPSCampaign ni sus métodos.
"""
import json
from pathlib import Path

import numpy as np

BEACONS = [f"b{i:02d}" for i in range(1, 9)]
RSSI_COLUMNS = [f"rssi_{b}" for b in BEACONS]
PRESENCE_COLUMNS = [f"present_{b}" for b in BEACONS]
MISSING_RSSI = -100.0
BEACON_MAPPING_FILE = (
    Path(__file__).resolve().parents[1] / "configs" / "tfg_beacon_mapping.json"
)


def normalize_mac(mac):
    if mac is None:
        return None
    return str(mac).strip().lower().replace("-", ":")


def load_beacon_mapping():
    with BEACON_MAPPING_FILE.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if set(mapping) != set(BEACONS):
        raise ValueError("El mapeo debe contener exactamente las balizas b01-b08")
    normalized = {beacon: normalize_mac(mapping[beacon]) for beacon in BEACONS}
    if None in normalized.values() or len(set(normalized.values())) != 8:
        raise ValueError("Las ocho MAC deben ser diferentes y no estar vacías")
    return normalized


def build_tfg_fingerprint(signals, beacon_mapping=None):
    """Convierte lecturas de UNA captura en [8 RSSI medianos, 8 máscaras].

    Cada lectura es {'Mac': ..., 'RSSI': ...} u objeto con estos atributos.
    El valor ausente es -100, pero su máscara distingue una lectura observada
    de RSSI -100 de la ausencia real de dicha baliza.
    """
    mapping = beacon_mapping if beacon_mapping is not None else load_beacon_mapping()
    mac_to_beacon = {normalize_mac(mac): b for b, mac in mapping.items()}
    readings = {beacon: [] for beacon in BEACONS}

    for signal in signals:
        mac = signal.get("Mac") if isinstance(signal, dict) else signal.Mac
        value = signal.get("RSSI") if isinstance(signal, dict) else signal.RSSI
        beacon = mac_to_beacon.get(normalize_mac(mac))
        if beacon is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            readings[beacon].append(value)

    rssis = [
        float(np.median(readings[b])) if readings[b] else MISSING_RSSI
        for b in BEACONS
    ]
    masks = [1.0 if readings[b] else 0.0 for b in BEACONS]
    return np.asarray([rssis + masks], dtype=float)


def build_tfg_campaign_matrix(campaign_id):
    """Carga capturas importadas de una campaña, sin fusionar orientaciones.

    La matriz incluye 8 RSSI, 8 máscaras y [x,y,z] (19 columnas).
    Usa TODAS las lecturas de cada captura, sin filtrar canales ni protocolos,
    como scripts/02_build_dataset.py. Solo descarta capturas sin señales BLE.
    Las coordenadas provienen de Capture tras la importación IPSCampaign.
    """
    from config import db
    from models.Capture import Capture
    from models.BeaconBleSignal import BeaconBleSignal

    rows = (
        db.session.query(
            Capture.Id,
            Capture.Position_x,
            Capture.Position_y,
            Capture.Position_z,
            BeaconBleSignal.Mac,
            BeaconBleSignal.RSSI,
        )
        .outerjoin(BeaconBleSignal, BeaconBleSignal.Id_capture == Capture.Id)
        .filter(Capture.Id_campaign == int(campaign_id))
        .order_by(Capture.Id.asc())
        .yield_per(2000)
    )

    mapping = load_beacon_mapping()
    matrix = []
    capture_ids = []
    current_id = None
    coordinates = None
    signals = []

    def append_capture():
        if not signals:
            return
        xy_z = np.asarray(coordinates, dtype=float)
        if not np.isfinite(xy_z).all():
            raise ValueError(f"Coordenadas no finitas en captura {current_id}")
        features = build_tfg_fingerprint(signals, mapping)[0]
        matrix.append(np.concatenate((features, xy_z)))
        capture_ids.append(int(current_id))

    for capture_id, x, y, z, mac, rssi in rows:
        if current_id is not None and capture_id != current_id:
            append_capture()
            signals = []
        if capture_id != current_id:
            current_id = capture_id
            coordinates = (x, y, z)
        if mac is not None:
            signals.append({"Mac": mac, "RSSI": rssi})

    if current_id is not None:
        append_capture()

    if not matrix:
        raise ValueError(f"La campaña {campaign_id} no contiene capturas BLE")
    return np.vstack(matrix), capture_ids
