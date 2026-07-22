"""Build the leakage-free multi-day IPSCampaign dataset.

Training uses odd-numbered physical points from the eligible April campaigns.
Evaluation uses even-numbered points from 16 April. Coordinates from every day
are normalized to the 16 April grid and only the seven intended iBeacon MACs
(all installed beacons except beacon 5) are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter
from pathlib import Path


BEACON_5_MAC = "f6:f8:05:65:70:30"
ALLOWED_MACS = (
    "d9:8f:81:7a:86:88",
    "dc:d7:d7:dd:ec:4a",
    "e8:36:9c:5d:bc:14",
    "e9:66:9f:26:a9:b9",
    "f0:31:71:50:65:ea",
    "f7:64:7d:4e:cd:6b",
    "fe:c4:be:cc:e2:05",
)
ROTATIONS = (0.0, 90.0, 180.0, 270.0)
MAX_POSITION_DISTANCE_METRES = 0.15

TRAINING_DATABASES = (
    ("2026-04-09", "prueba_9_4_merged.sqlite3", "46121"),
    ("2026-04-10", "toma_datos_10_4.sqlite3", "46122"),
    ("2026-04-13", "toma_datos_13_4.sqlite3", "46125"),
    # The 16/4 campaign is complete and is compatible after removing beacon 5.
    ("2026-04-16", "toma_datos_16_4.sqlite3", "46128"),
    ("2026-04-17", "toma_datos_17_4.sqlite3", "46129"),
    ("2026-04-20", "toma_datos_20_4.sqlite3", "46132"),
    ("2026-04-22", "toma_datos_22_4.sqlite3", "46134"),
)

OUTPUT_DIRECTORY = (
    "paquete_IPSCampaign_multicampana_impares_"
    "eval16_4_pares_sin_beacon5"
)
REFERENCE_DATABASE = "referencia_multicampana_impares_sin_beacon5.sqlite3"
EVALUATION_DATABASE = "aleatoria_16_4_pares_sin_beacon5.sqlite3"


def normalize_mac(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", ":")


def ordered_positions(connection: sqlite3.Connection) -> list[tuple[float, float, float]]:
    rows = connection.execute(
        """
        SELECT Position_x, Position_y, Position_z, MIN(Id)
        FROM Capture
        GROUP BY Position_x, Position_y, Position_z
        ORDER BY MIN(Id)
        """
    )
    return [(float(x), float(y), float(z)) for x, y, z, _ in rows]


def load_canonical_grid(database: Path) -> list[tuple[float, float, float]]:
    with sqlite3.connect(database) as connection:
        grid = ordered_positions(connection)
    if len(grid) != 60:
        raise ValueError(f"Expected 60 canonical points, found {len(grid)}")
    return grid


def map_positions(
    positions: list[tuple[float, float, float]],
    canonical: list[tuple[float, float, float]],
) -> tuple[dict[tuple[float, float, float], int], list[dict]]:
    mapping: dict[tuple[float, float, float], int] = {}
    excluded: list[dict] = []
    assigned: set[int] = set()

    for source_sequence, position in enumerate(positions, start=1):
        distances = [math.dist(position, target) for target in canonical]
        distance = min(distances)
        point_number = distances.index(distance) + 1

        if distance > MAX_POSITION_DISTANCE_METRES:
            excluded.append(
                {
                    "source_sequence": source_sequence,
                    "position": list(position),
                    "nearest_point": point_number,
                    "distance_metres": round(distance, 6),
                    "reason": "outside_position_tolerance",
                }
            )
            continue

        if point_number in assigned:
            raise ValueError(
                f"Two source positions map to canonical point {point_number}"
            )
        assigned.add(point_number)
        mapping[position] = point_number

    return mapping, excluded


def allowed_signal_counts(connection: sqlite3.Connection) -> dict[int, int]:
    placeholders = ",".join("?" for _ in ALLOWED_MACS)
    return dict(
        connection.execute(
            f"""
            SELECT Id_capture, COUNT(*)
            FROM Beacon_BLE_Signal
            WHERE LOWER(REPLACE(TRIM(Mac), '-', ':')) IN ({placeholders})
              AND LOWER(TRIM(Protocol)) = 'ibeacon'
              AND Channel IN (37, 38, 39)
            GROUP BY Id_capture
            """,
            ALLOWED_MACS,
        )
    )


def collect_captures(
    database: Path,
    source_index: int,
    source_date: str,
    workbook_serial: str,
    canonical: list[tuple[float, float, float]],
    parity: int,
) -> tuple[list[dict], dict]:
    with sqlite3.connect(database) as connection:
        positions = ordered_positions(connection)
        position_mapping, excluded = map_positions(positions, canonical)
        signal_counts = allowed_signal_counts(connection)
        captures = connection.execute(
            """
            SELECT Id, Date, Light, Temperature, Relative_humidity,
                   Absolute_humidity, Position_x, Position_y, Position_z,
                   Platform_angle, Dongle_rotation
            FROM Capture
            ORDER BY Id
            """
        ).fetchall()

    records: list[dict] = []
    empty_captures = 0
    for row in captures:
        position = (float(row[6]), float(row[7]), float(row[8]))
        point_number = position_mapping.get(position)
        if point_number is None or point_number % 2 != parity:
            continue
        rotation = float(row[10])
        if rotation not in ROTATIONS:
            continue
        if signal_counts.get(int(row[0]), 0) == 0:
            empty_captures += 1
            continue

        records.append(
            {
                "source_index": source_index,
                "source_date": source_date,
                "source_database": database.name,
                "source_capture_id": int(row[0]),
                "point_number": point_number,
                "rotation": rotation,
                "capture": (
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    canonical[point_number - 1][0],
                    canonical[point_number - 1][1],
                    canonical[point_number - 1][2],
                    row[9],
                    row[10],
                ),
            }
        )

    point_numbers = sorted({record["point_number"] for record in records})
    rotations = Counter(record["rotation"] for record in records)
    expected = set(range(1 if parity else 2, 61, 2))
    audit = {
        "date": source_date,
        "workbook_serial": workbook_serial,
        "database": database.name,
        "source_captures": len(captures),
        "source_positions": len(positions),
        "mapped_positions": len(position_mapping),
        "selected_points": point_numbers,
        "missing_selected_points": sorted(expected - set(point_numbers)),
        "selected_captures": len(records),
        "captures_by_rotation": {
            str(int(rotation)): rotations.get(rotation, 0) for rotation in ROTATIONS
        },
        "empty_captures_discarded": empty_captures,
        "excluded_positions": excluded,
    }
    return records, audit


def prepare_empty_database(template: Path, destination: Path) -> None:
    shutil.copy2(template, destination)
    with sqlite3.connect(destination) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        for table in (
            "Beacon_BLE_Signal",
            "Capture",
            "Beacon_Configuration",
            "Dongle_Receptor",
            "Alert",
        ):
            connection.execute(f'DELETE FROM "{table}"')
        connection.execute("DELETE FROM sqlite_sequence")
        connection.commit()


def copy_static_metadata(template: Path, destination: Path) -> None:
    with sqlite3.connect(template) as source, sqlite3.connect(destination) as target:
        dongles = source.execute("SELECT Id, name FROM Dongle_Receptor ORDER BY Id")
        target.executemany("INSERT INTO Dongle_Receptor (Id, name) VALUES (?, ?)", dongles)

        configurations = source.execute(
            """
            SELECT Id, Mac, Power, Calibrated_power, Update_frequency, Identificator
            FROM Beacon_Configuration
            ORDER BY Id
            """
        )
        kept_configurations = [
            row for row in configurations if normalize_mac(row[1]) in ALLOWED_MACS
        ]
        target.executemany(
            """
            INSERT INTO Beacon_Configuration
                (Id, Mac, Power, Calibrated_power, Update_frequency, Identificator)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            kept_configurations,
        )
        target.commit()


def insert_captures(destination: Path, records: list[dict]) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    records.sort(
        key=lambda record: (
            record["point_number"],
            record["source_index"],
            record["rotation"],
            record["source_capture_id"],
        )
    )
    with sqlite3.connect(destination) as connection:
        for record in records:
            cursor = connection.execute(
                """
                INSERT INTO Capture
                    (Date, Light, Temperature, Relative_humidity,
                     Absolute_humidity, Position_x, Position_y, Position_z,
                     Platform_angle, Dongle_rotation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                record["capture"],
            )
            mapping[(record["source_index"], record["source_capture_id"])] = (
                cursor.lastrowid
            )
        connection.commit()
    return mapping


def copy_signals(
    sources: list[Path],
    destination: Path,
    capture_mapping: dict[tuple[int, int], int],
) -> None:
    placeholders = ",".join("?" for _ in ALLOWED_MACS)
    insert_sql = """
        INSERT INTO Beacon_BLE_Signal
            (Id_capture, N_reading, Date_hour, Mac, Pack_size, Channel, RSSI,
             PDU_type, CRC, Protocol, Identificator)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(destination) as target:
        for source_index, source_path in enumerate(sources):
            source_capture_ids = sorted(
                source_capture_id
                for index, source_capture_id in capture_mapping
                if index == source_index
            )
            if not source_capture_ids:
                continue
            capture_placeholders = ",".join("?" for _ in source_capture_ids)
            with sqlite3.connect(source_path) as source:
                rows = source.execute(
                    f"""
                    SELECT Id_capture, N_reading, Date_hour, Mac, Pack_size,
                           Channel, RSSI, PDU_type, CRC, Protocol, Identificator
                    FROM Beacon_BLE_Signal
                    WHERE Id_capture IN ({capture_placeholders})
                      AND LOWER(REPLACE(TRIM(Mac), '-', ':')) IN ({placeholders})
                      AND LOWER(TRIM(Protocol)) = 'ibeacon'
                      AND Channel IN (37, 38, 39)
                    ORDER BY Id
                    """,
                    (*source_capture_ids, *ALLOWED_MACS),
                )
                batch = []
                for row in rows:
                    new_capture_id = capture_mapping[(source_index, int(row[0]))]
                    batch.append((new_capture_id, *row[1:]))
                    if len(batch) == 5000:
                        target.executemany(insert_sql, batch)
                        batch.clear()
                if batch:
                    target.executemany(insert_sql, batch)
            target.commit()


def build_database(
    template: Path,
    sources: list[Path],
    destination: Path,
    records: list[dict],
) -> None:
    prepare_empty_database(template, destination)
    copy_static_metadata(template, destination)
    mapping = insert_captures(destination, records)
    copy_signals(sources, destination, mapping)
    with sqlite3.connect(destination) as connection:
        connection.execute("VACUUM")


def database_stats(database: Path) -> dict:
    with sqlite3.connect(database) as connection:
        captures = connection.execute("SELECT COUNT(*) FROM Capture").fetchone()[0]
        positions = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT 1 FROM Capture
                GROUP BY Position_x, Position_y, Position_z
            )
            """
        ).fetchone()[0]
        signals = connection.execute(
            "SELECT COUNT(*) FROM Beacon_BLE_Signal"
        ).fetchone()[0]
        macs = [
            row[0]
            for row in connection.execute(
                """
                SELECT DISTINCT LOWER(REPLACE(TRIM(Mac), '-', ':'))
                FROM Beacon_BLE_Signal
                ORDER BY 1
                """
            )
        ]
        channels = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT Channel FROM Beacon_BLE_Signal ORDER BY Channel"
            )
        ]
        rotations = {
            str(int(rotation)): count
            for rotation, count in connection.execute(
                """
                SELECT Dongle_rotation, COUNT(*)
                FROM Capture
                GROUP BY Dongle_rotation
                ORDER BY Dongle_rotation
                """
            )
        }
        empty_captures = connection.execute(
            """
            SELECT COUNT(*)
            FROM Capture c
            LEFT JOIN Beacon_BLE_Signal s ON s.Id_capture = c.Id
            WHERE s.Id IS NULL
            """
        ).fetchone()[0]
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()

    return {
        "captures": captures,
        "positions": positions,
        "signals": signals,
        "macs": macs,
        "channels": channels,
        "captures_by_rotation": rotations,
        "empty_captures": empty_captures,
        "integrity_check": integrity,
        "foreign_key_violations": len(foreign_keys),
    }


def write_general_config(path: Path) -> None:
    path.write_text(
        """[BLE]
Port = 9000
Address = localhost
LogFile = ./logs/logBLE.txt
Duration = 15

[Orientator]
Port = 32444
Address = localhost
LogFile = ./logs/logOrientador.txt
Rotations = 0,90,180,270

[Weather]
Port = 32443
Address = localhost
LogFile = ./logs/logWeather.txt

[Robo Map]
Port = 1998
Address = localhost
LogFile = ./logs/logRoboMap.txt
""",
        encoding="utf-8",
    )


def write_points_config(
    path: Path,
    canonical: list[tuple[float, float, float]],
    parity: int,
) -> None:
    points = [
        point for point_number, point in enumerate(canonical, start=1)
        if point_number % 2 == parity
    ]
    sections = []
    for index, (x, y, z) in enumerate(points):
        sections.append(
            f"[POINT_{index}]\n"
            f"X = {x:.6f}\nY = {y:.6f}\nZ = {z:g}\n"
            "Rotations = 0.0,90.0,180.0,270.0\n"
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_dataset(reference: dict, evaluation: dict, overlap: set) -> None:
    expected_macs = list(ALLOWED_MACS)
    for label, stats in (("reference", reference), ("evaluation", evaluation)):
        if stats["positions"] != 30:
            raise ValueError(f"{label} has {stats['positions']} points instead of 30")
        if stats["macs"] != expected_macs:
            raise ValueError(f"{label} MACs differ from the intended seven beacons")
        if stats["channels"] != [37, 38, 39]:
            raise ValueError(f"{label} does not contain exactly channels 37, 38, 39")
        if stats["empty_captures"]:
            raise ValueError(f"{label} contains empty captures")
        if stats["integrity_check"] != "ok" or stats["foreign_key_violations"]:
            raise ValueError(f"{label} failed SQLite integrity checks")
    if evaluation["captures"] != 120:
        raise ValueError("Evaluation must contain 30 points x 4 orientations")
    if overlap:
        raise ValueError("Training and evaluation coordinates overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    campaigns = workspace / "Campañas"
    output = campaigns / OUTPUT_DIRECTORY
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Remove it explicitly before rebuilding."
        )
    output.mkdir(parents=True)

    canonical_database = campaigns / "toma_datos_16_4.sqlite3"
    canonical = load_canonical_grid(canonical_database)
    training_paths = [campaigns / name for _, name, _ in TRAINING_DATABASES]

    training_records: list[dict] = []
    source_audits = []
    for source_index, (date, name, serial) in enumerate(TRAINING_DATABASES):
        records, audit = collect_captures(
            campaigns / name,
            source_index,
            date,
            serial,
            canonical,
            parity=1,
        )
        training_records.extend(records)
        source_audits.append(audit)

    evaluation_records, evaluation_source_audit = collect_captures(
        canonical_database,
        0,
        "2026-04-16",
        "46128",
        canonical,
        parity=0,
    )

    reference_path = output / REFERENCE_DATABASE
    evaluation_path = output / EVALUATION_DATABASE
    build_database(
        canonical_database,
        training_paths,
        reference_path,
        training_records,
    )
    build_database(
        canonical_database,
        [canonical_database],
        evaluation_path,
        evaluation_records,
    )

    general_config = output / "01_Config.conf"
    evaluation_config = output / "02_Config_Ale_Points.conf"
    reference_config = output / "03_Config_Ref_Points.conf"
    write_general_config(general_config)
    write_points_config(evaluation_config, canonical, parity=0)
    write_points_config(reference_config, canonical, parity=1)

    reference_stats = database_stats(reference_path)
    evaluation_stats = database_stats(evaluation_path)
    reference_positions = {tuple(training_records[index]["capture"][5:8]) for index in range(len(training_records))}
    evaluation_positions = {tuple(record["capture"][5:8]) for record in evaluation_records}
    overlap = reference_positions & evaluation_positions
    assert_dataset(reference_stats, evaluation_stats, overlap)

    manifest = {
        "purpose": (
            "Train on odd physical points from eligible campaigns and evaluate "
            "on even physical points from 16/4 without beacon 5"
        ),
        "workbook": "RESUMEN_datos_faltan (2).xlsx",
        "selection": {
            "included_rows_with_only_beacon_5_missing": [
                "46121", "46122", "46125", "46129", "46132", "46134"
            ],
            "included_complete_row": "46128",
            "excluded_rows": {
                "46135": "beacons 3 and 5 affected",
                "46136": "beacons 3 and 5 affected",
            },
        },
        "beacon_5": {
            "position": [6.52, 9.33, 2.0],
            "mac": BEACON_5_MAC,
            "excluded_from_training_and_evaluation": True,
        },
        "allowed_macs": list(ALLOWED_MACS),
        "channels": [37, 38, 39],
        "orientations": list(ROTATIONS),
        "coordinate_reference": "toma_datos_16_4.sqlite3",
        "position_tolerance_metres": MAX_POSITION_DISTANCE_METRES,
        "training": {
            "physical_points": "odd numbers 1,3,...,59",
            "source_audits": source_audits,
            "database": REFERENCE_DATABASE,
            "stats": reference_stats,
        },
        "evaluation": {
            "physical_points": "even numbers 2,4,...,60",
            "source_audit": evaluation_source_audit,
            "database": EVALUATION_DATABASE,
            "stats": evaluation_stats,
        },
        "leakage_check": {
            "overlapping_coordinates": len(overlap),
            "result": "ok" if not overlap else "failed",
        },
        "orientation_result": (
            "The four predictions for each evaluation point are fused by the "
            "application using the coordinate-wise median."
        ),
    }
    manifest_path = output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    readme = output / "README_IMPORTACION.md"
    readme.write_text(
        f"""# Campaña multijornada sin beacon 5

Este paquete entrena con los puntos físicos impares de 9/4, 10/4, 13/4,
16/4, 17/4, 20/4 y 22/4. Evalúa exclusivamente los 30 puntos pares del
16/4. Las coordenadas se han normalizado contra la cuadrícula del 16/4 y no
hay ningún punto físico compartido entre entrenamiento y evaluación.

Se conservan los canales 37, 38 y 39, las orientaciones 0, 90, 180 y 270, y
los siete beacons instalados distintos del beacon 5
(`{BEACON_5_MAC}`, posición `(6.52, 9.33, 2.0)`).

## Archivos que debes elegir en IPSCampaign

- Base aleatoria: `{EVALUATION_DATABASE}`
- Base de referencia: `{REFERENCE_DATABASE}`
- Configuraciones, en este orden:
  1. `01_Config.conf`
  2. `02_Config_Ale_Points.conf`
  3. `03_Config_Ref_Points.conf`

Nombre recomendado: `DB_MULTI_IMPARES_REF_16_4_PARES_EVAL_SIN_B5`.

`MANIFEST.json` contiene la auditoría por jornada, las capturas descartadas,
los conteos finales y la comprobación de ausencia de fuga espacial.
""",
        encoding="utf-8",
    )

    hash_files = (
        general_config,
        evaluation_config,
        reference_config,
        evaluation_path,
        reference_path,
        manifest_path,
        readme,
    )
    (output / "MANIFESTO_SHA256.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in hash_files),
        encoding="utf-8",
    )

    print(json.dumps({"output": str(output), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
