"""Prepara el escenario PR del 16/04 sin utilizar los puntos PA finales.

Uso desde la raíz de indoor-positioning-deep-learning:
    python IPSCM/experiments/prepare_tfg_same_day_campaign.py \
        --source ../tfg-ips/data/raw/toma_datos_16_4.sqlite3 \
        --output Campañas/TFG_16_4

Genera bases referencia (puntos impares) / evaluación (pares), tres .ini y
un manifiesto de reproducibilidad. No modifica la SQLite original.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def read_grid(connection):
    coordinates = connection.execute(
        """SELECT Position_x, Position_y, Position_z, MIN(Id)
        FROM Capture GROUP BY Position_x, Position_y, Position_z
        ORDER BY MIN(Id)"""
    ).fetchall()
    if len(coordinates) != 60:
        raise ValueError(
            f"Se esperaban 60 posiciones en la campaña del 16/04; encontradas: {len(coordinates)}"
        )
    return [tuple(map(float, row[:3])) for row in coordinates]


def collect_captures(connection, grid):
    point_by_coord = {coordinate: index + 1 for index, coordinate in enumerate(grid)}
    captures = connection.execute(
        """SELECT Id, Position_x, Position_y, Position_z, Dongle_rotation
        FROM Capture ORDER BY Id"""
    ).fetchall()
    training, evaluation = [], []
    orientations = defaultdict(set)
    for capture_id, x, y, z, rotation in captures:
        point = point_by_coord.get((float(x), float(y), float(z)))
        if point is None:
            raise ValueError(f"Captura {capture_id} no está en la cuadrícula canónica")
        orientations[point].add(float(rotation))
        (training if point % 2 == 1 else evaluation).append(int(capture_id))
    if not training or not evaluation:
        raise ValueError("No se encontraron capturas para ambas particiones")
    if set(training).intersection(evaluation):
        raise ValueError("Las particiones comparten capturas")
    return training, evaluation, orientations


def build_sqlite(source_connection, output_path, allowed_capture_ids):
    if output_path.exists():
        raise FileExistsError(
            f"Ya existe {output_path}; usa una carpeta de salida nueva para no sobrescribir datos"
        )
    destination = sqlite3.connect(output_path)
    try:
        source_connection.backup(destination)
        destination.execute("PRAGMA foreign_keys=OFF")
        placeholders = ",".join("?" for _ in allowed_capture_ids)
        destination.execute(
            f"DELETE FROM Beacon_BLE_Signal WHERE Id_capture NOT IN ({placeholders})",
            allowed_capture_ids,
        )
        destination.execute(
            f"DELETE FROM Capture WHERE Id NOT IN ({placeholders})",
            allowed_capture_ids,
        )
        destination.commit()
        destination.execute("VACUUM")
        kept_captures = destination.execute("SELECT COUNT(*) FROM Capture").fetchone()[0]
        if kept_captures != len(allowed_capture_ids):
            raise ValueError("El número de capturas exportadas no coincide con el manifiesto")
    finally:
        destination.close()


def write_config(path, points, grid, orientations):
    lines = []
    for order, point in enumerate(points):
        x, y, z = grid[point - 1]
        rotations = sorted(orientations[point])
        lines.extend([
            f"[POINT_{order}]",
            f"X = {repr(x)}",
            f"Y = {repr(y)}",
            f"Z = {repr(z)}",
            "Rotations = " + ", ".join(repr(rotation) for rotation in rotations),
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("../tfg-ips/data/raw/toma_datos_16_4.sqlite3"),
    )
    parser.add_argument("--output", type=Path, default=Path("Campañas/TFG_16_4"))
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(
        args.source.resolve().as_uri() + "?mode=ro", uri=True
    ) as source:
        grid = read_grid(source)
        training, evaluation, orientations = collect_captures(source, grid)
        ref_path = args.output / "tfg_reference_odd.sqlite3"
        eval_path = args.output / "tfg_evaluation_even.sqlite3"
        build_sqlite(source, ref_path, training)
        build_sqlite(source, eval_path, evaluation)

    odd_points = list(range(1, 61, 2))
    even_points = list(range(2, 61, 2))
    write_config(args.output / "ref_points.ini", odd_points, grid, orientations)
    write_config(args.output / "ale_points.ini", even_points, grid, orientations)
    (args.output / "ble.ini").write_text(
        "[BLE]\nDuration = 0\n\n[Orientator]\nRotations = 4\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol": "toma_datos_16_4_PR_odd_train_even_test",
        "source": str(args.source.resolve()),
        "sha256_source": checksum(args.source),
        "reference_capture_count": len(training),
        "evaluation_capture_count": len(evaluation),
        "reference_points": odd_points,
        "evaluation_points": even_points,
        "n_physical_points": len(grid),
        "reference_database": str(ref_path),
        "evaluation_database": str(eval_path),
        "note": "PR-only; PA final no se utiliza. Duration=0 es metadato de importación.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
