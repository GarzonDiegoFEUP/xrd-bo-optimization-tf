"""Synthetic tests for the four-phase Le Bail metrics export."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from xrd_bo_optimization_tf.rietx_refined_metrics import (
    OUTPUT_COLUMNS,
    REQUIRED_COLUMNS,
    export_refined_metrics,
)


def sample(**changes):
    row = {
        "sample_id": "sample-1", "status": "converged", "rwp": "0.19",
        "omega_deg": "0.2", "phase_count": "4",
        "phase_labels": "α-BaCu₂Se₂;β-BaCu₂Se₂;BaSeO₃;BaSe",
        "diagnostic_codes": "WARNING;WARNING;STAGE_MAX_ITER",
        "fit_plot": "outputs/example.png",
        "warning_geometry": "Bragg-Brentano proxy; no fixed-incidence GIXRD",
    }
    row.update(changes)
    return row


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "summary.csv"
        self.output = Path(self.tmp.name) / "out" / "metrics.csv"

    def write_source(self, rows, columns=REQUIRED_COLUMNS):
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def read_output(self):
        with self.output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, list(OUTPUT_COLUMNS))
            return list(reader)

    def test_preserves_rows_status_observations_and_null_reasons(self):
        self.write_source([
            sample(),
            sample(sample_id="sample-2", status="max_iter", rwp="0.21000", diagnostic_codes="A;A;B"),
            sample(sample_id="sample-3", status="error", rwp="", omega_deg="",
                   phase_count="", phase_labels="", diagnostic_codes="", fit_plot="", warning_geometry=""),
        ])
        self.assertEqual(export_refined_metrics(self.source, self.output), 3)
        rows = self.read_output()
        self.assertEqual([r["sample_id"] for r in rows], ["sample-1", "sample-2", "sample-3"])
        self.assertEqual([r["status"] for r in rows], ["converged", "max_iter", "error"])
        self.assertEqual(rows[1]["rwp"], "0.21000")
        self.assertEqual(rows[1]["diagnostic_codes"], "A;A;B")
        self.assertEqual(rows[0]["fit_geometry_warning"], sample()["warning_geometry"])
        self.assertEqual(rows[0]["phase_labels"], sample()["phase_labels"])
        self.assertEqual(rows[0]["fit_plot"], sample()["fit_plot"])
        self.assertEqual(rows[2]["rwp"], "")
        for row in rows:
            self.assertEqual(row["source_summary"], str(self.source))
            self.assertEqual(row["source_summary_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
            for value, reason in (
                ("alpha_weight_fraction", "alpha_weight_fraction_availability_reason"),
                ("alpha_crystallite_size_nm", "alpha_crystallite_size_availability_reason"),
                ("alpha_microstrain", "alpha_microstrain_availability_reason"),
            ):
                self.assertEqual(row[value], "")
                self.assertTrue(row[reason])

    def assert_rejected_without_overwrite(self, pattern):
        self.output.parent.mkdir(exist_ok=True)
        self.output.write_text("existing output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, pattern):
            export_refined_metrics(self.source, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output")

    def test_missing_column(self):
        self.write_source([sample()], [c for c in REQUIRED_COLUMNS if c != "rwp"])
        self.assert_rejected_without_overwrite("Missing required columns: rwp")

    def test_duplicate_sample_including_whitespace(self):
        self.write_source([sample(), sample(sample_id=" sample-1 ")])
        self.assert_rejected_without_overwrite("Duplicate sample_id")

    def test_invalid_rows(self):
        cases = [
            (sample(sample_id=" "), "Missing sample_id"),
            (sample(status=""), "missing status"),
            (sample(rwp="not a number"), "invalid rwp"),
            (sample(rwp="nan"), "invalid rwp"),
            (sample(rwp="-0.1"), "invalid rwp"),
            (sample(omega_deg="inf"), "invalid omega_deg"),
            (sample(phase_count="3"), "expected four phases"),
            (sample(phase_labels="α-BaCu₂Se₂;β-BaCu₂Se₂;Cu₂Se;BaSe"), "unexpected four-phase labels"),
        ]
        for row, pattern in cases:
            with self.subTest(row=row):
                self.write_source([sample(sample_id="good"), row])
                self.assert_rejected_without_overwrite(pattern)

    def test_empty_summary_and_duplicate_header(self):
        self.write_source([])
        self.assert_rejected_without_overwrite("contains no rows")
        self.write_source([sample()], (*REQUIRED_COLUMNS, "rwp"))
        self.assert_rejected_without_overwrite("Duplicate input column")

    def test_malformed_row(self):
        self.write_source([sample()])
        with self.source.open("a", encoding="utf-8") as handle:
            handle.write("too,few,columns\n")
        self.assert_rejected_without_overwrite("Malformed CSV row")

    def test_input_cannot_be_overwritten(self):
        self.write_source([sample()])
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "must differ"):
            export_refined_metrics(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            export_refined_metrics(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()