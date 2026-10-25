import configparser
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.prepare_tfg_same_day_campaign import (
    build_sqlite,
    collect_captures,
    read_grid,
    write_config,
)


class PrepareTFGSameDayTest(unittest.TestCase):
    def test_grid_parity_and_sqlite_subset_preserve_capture_signal_alignment(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = Path(tempdir)
            original = directory / "input.sqlite3"
            with sqlite3.connect(original) as connection:
                connection.executescript("""
                    CREATE TABLE Capture (
                        Id INTEGER PRIMARY KEY,
                        Position_x REAL,
                        Position_y REAL,
                        Position_z REAL,
                        Dongle_rotation REAL
                    );
                    CREATE TABLE Beacon_BLE_Signal (
                        Id INTEGER PRIMARY KEY,
                        Id_capture INTEGER,
                        Mac TEXT,
                        RSSI INTEGER
                    );
                """)
                capture_id = 0
                for point in range(60):
                    for rotation in (0.0, 90.0):
                        capture_id += 1
                        connection.execute(
                            "INSERT INTO Capture VALUES (?, ?, ?, ?, ?)",
                            (capture_id, float(point), 0.0, 1.2, rotation)
                        )
                        connection.execute(
                            "INSERT INTO Beacon_BLE_Signal VALUES (?, ?, ?, ?)",
                            (capture_id, capture_id, "aa:bb:cc:dd:ee:ff", -70)
                        )
                connection.commit()

            with sqlite3.connect(original) as source:
                grid = read_grid(source)
                reference, evaluation, rotations = collect_captures(source, grid)
                self.assertEqual(len(reference), 60)
                self.assertEqual(len(evaluation), 60)
                self.assertFalse(set(reference) & set(evaluation))
                output = directory / "ref.sqlite3"
                build_sqlite(source, output, reference)

            with sqlite3.connect(output) as ref:
                self.assertEqual(ref.execute("SELECT COUNT(*) FROM Capture").fetchone()[0], 60)
                self.assertEqual(ref.execute(
                    "SELECT COUNT(*) FROM Beacon_BLE_Signal"
                ).fetchone()[0], 60)
                self.assertEqual(ref.execute("""
                    SELECT COUNT(*) FROM Beacon_BLE_Signal
                    WHERE Id_capture NOT IN (SELECT Id FROM Capture)
                """).fetchone()[0], 0)

            config_path = directory / "ref_points.ini"
            write_config(config_path, list(range(1, 61, 2)), grid, rotations)
            config = configparser.ConfigParser()
            config.read(config_path, encoding="utf-8")
            self.assertEqual(len(config.sections()), 30)
            self.assertEqual(
                config["POINT_0"]["Rotations"],
                "0.0, 90.0",
            )


if __name__ == "__main__":
    unittest.main()
