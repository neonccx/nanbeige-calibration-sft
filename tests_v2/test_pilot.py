import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibration_v2.s21 import fit, measure, notch
from calibration_v2.build_pilot import analyze_artifact, build


class S21Tests(unittest.TestCase):
    def test_notch_limit(self):
        self.assertAlmostEqual(notch([6.5e9], 6.5e9, 4000, 0.6)[0].real, 0.4)
        self.assertAlmostEqual(notch([6.5e9], 6.5e9, 4000, 0.6)[0].imag, 0)

    def test_recovery(self):
        for noise in (0, 0.002, 0.01):
            obs = measure(6.502e9, 4500, 0.6, center_hz=6.5e9, span_hz=24e6, seed=5, noise_std=noise)
            result = fit(obs)
            self.assertTrue(result["reliable"])
            self.assertLess(abs(result["fit"]["resonance_hz"] - 6.502e9), 2e4)
            self.assertLess(abs(result["fit"]["loaded_q"] / 4500 - 1), 0.04)

    def test_edge_rejected(self):
        obs = measure(6.501e9, 4000, 0.6, center_hz=6.488e9, span_hz=28e6, seed=3)
        result = fit(obs)
        self.assertFalse(result["reliable"])
        self.assertFalse(result["diagnostics"]["checks"]["not_at_scan_edge"])

    def test_invalid_and_unsupported_data(self):
        obs = measure(6.5e9, 4000, 0.6, center_hz=6.5e9, span_hz=24e6, seed=3)
        obs["q"][0] = float("nan")
        with self.assertRaises(ValueError):
            fit(obs)
        with self.assertRaises(ValueError):
            fit({"model": "uncalibrated_complex_response"})
        with self.assertRaises(ValueError):
            measure(6.5e9, 4000, 0.6, center_hz=6.5, span_hz=24e6, seed=3)

    def test_flat_trace_rejected(self):
        obs = {"model": "calibrated_ideal_notch", "frequency_hz": np.linspace(6.4e9, 6.6e9, 100).tolist(),
               "i": [1] * 100, "q": [0] * 100}
        self.assertFalse(fit(obs)["reliable"])

    def test_pilot_integrity_and_grouped_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            manifest = build(root, devices=5)
            self.assertFalse(manifest["training_ready"])
            self.assertEqual(sum(manifest["counts"].values()), 10)
            groups = []
            retries = 0
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
                groups.append({row["device_id"] for row in rows})
                for row in rows:
                    serialized = json.dumps(row["messages"], ensure_ascii=False)
                    self.assertTrue(serialized.isascii())
                    self.assertNotIn("evaluator_only", serialized)
                    pending = set()
                    for message in row["messages"]:
                        for call in message.get("tool_calls", []):
                            self.assertNotIn(call["id"], pending)
                            pending.add(call["id"])
                        if message["role"] == "tool":
                            self.assertIn(message["tool_call_id"], pending)
                            pending.remove(message["tool_call_id"])
                    self.assertFalse(pending)
                    retries += len(row["messages"]) > 7
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            self.assertGreater(retries, 0)
            path = next((root / "artifacts").rglob("*.json"))
            relative = str(path.relative_to(root))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn("reliable", analyze_artifact(root, relative, digest))
            with self.assertRaises(ValueError):
                analyze_artifact(root, relative, "0" * 64)
            with self.assertRaises(ValueError):
                analyze_artifact(root, "../outside.json", digest)
            with self.assertRaises(FileExistsError):
                build(root, devices=5)


if __name__ == "__main__":
    unittest.main()
