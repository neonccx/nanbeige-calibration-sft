"""Execute small S21 tool-use episodes. Never starts training or real instruments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .s21 import VERSION, fit, measure

TOOLS = [
    {"type": "function", "function": {"name": "experiment.s21",
     "description": "Acquire a bounded simulated calibrated-notch frequency scan; returns an artifact reference.",
     "parameters": {"type": "object", "additionalProperties": False,
       "properties": {"center_hz": {"type": "number", "minimum": 6e9, "maximum": 7e9},
                      "span_hz": {"type": "number", "minimum": 1e6, "maximum": 100e6}},
       "required": ["center_hz", "span_hz"]}}},
    {"type": "function", "function": {"name": "analysis.fit_s21",
     "description": "Fit a saved calibrated-notch artifact; report numerical estimates and scan quality.",
     "parameters": {"type": "object", "additionalProperties": False,
       "properties": {"artifact_path": {"type": "string"}, "sha256": {"type": "string"}},
       "required": ["artifact_path", "sha256"]}}},
]

SYSTEM = ("You plan simulated single-qubit calibration in English. Call numerical analysis tools; "
          "do not calculate fitted parameters yourself. Use only bounded registered tools. "
          "Use observed quality to choose a rescan or propose the next stage. "
          "Do not claim full calibration success from S21. Maximum two S21 scans. "
          "Never access hidden device truth or execute arbitrary code.")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n")


def call_message(call_id: str, name: str, args: dict, reason: str) -> dict:
    return {"role": "assistant", "content": reason,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def analyze_artifact(root: Path, relative: str, digest: str) -> dict:
    path = (root / relative).resolve()
    if not path.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Artifact must be inside the pilot artifact directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Artifact hash mismatch")
    return fit(json.loads(raw))


def build(output: Path, devices: int = 12) -> dict:
    if devices < 5:
        raise ValueError("At least five devices are required for three grouped splits")
    output.mkdir(parents=True, exist_ok=False)
    buckets = {name: [] for name in ("train", "validation", "test")}
    # Fixed device assignment BEFORE trajectory or prefix expansion.
    order = np.random.default_rng(20260904).permutation(devices).tolist()
    n_validation = max(1, devices // 5)
    n_test = max(1, devices // 5)
    assignment = {d: ("test" if index < n_test else "validation" if index < n_test + n_validation else "train")
                  for index, d in enumerate(order)}
    for device in range(devices):
        rng = np.random.default_rng(70000 + device)
        truth = {"resonance_hz": float(6.5e9 + rng.uniform(-2e6, 2e6)),
                 "loaded_q": float(rng.uniform(3000, 6000)), "depth": float(rng.uniform(0.35, 0.7))}
        for scenario in ("covered", "edge"):
            # Artifact names do not disclose the scenario/target label to the model.
            episode_id = hashlib.sha256(f"pilot:{device}:{scenario}".encode()).hexdigest()[:16]
            # Independent acquisition start, not a scan centered on hidden truth.
            center, span = (6.5e9, 24e6) if scenario == "covered" else (6.488e9, 28e6)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                f"Start the S21 stage using center_hz={center} and span_hz={span}. "
                "Analyze the saved measurement, retry if quality is inadequate, and recommend the next stage. "
                "This is an ideal calibrated-notch simulation, not a real device."}]
            trace = []
            for step in range(2):
                args = {"center_hz": center, "span_hz": span}
                cid = f"{episode_id}_measure_{step}"
                messages.append(call_message(cid, "experiment.s21", args,
                    "Acquire the requested scan." if step == 0 else
                    "The numerical quality checks rejected the previous scan. Increase coverage within the scan budget."))
                obs = measure(**truth, **args, seed=80000 + device * 10 + step + (5 if scenario == "edge" else 0))
                relative = f"artifacts/{episode_id}/scan_{step}.json"
                path = output / relative
                dump(path, obs)
                reference = {"artifact_path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                messages.append({"role": "tool", "tool_call_id": cid, "name": "experiment.s21",
                                 "content": json.dumps(reference | {"synthetic": True, "points": 241})})
                aid = f"{episode_id}_analysis_{step}"
                messages.append(call_message(aid, "analysis.fit_s21", reference,
                                             "Use the numerical fitter to estimate resonance and check coverage."))
                analysis = analyze_artifact(output, relative, reference["sha256"])
                messages.append({"role": "tool", "tool_call_id": aid, "name": "analysis.fit_s21",
                                 "content": json.dumps(analysis, allow_nan=False)})
                trace.append({"artifact": reference, "analysis": analysis})
                if analysis["reliable"]:
                    break
                # Policy sees only the tool result; latent parameters never enter this branch.
                center = float(np.clip(analysis["fit"].get("resonance_hz", center), 6e9, 7e9))
                span = min(100e6, 2 * span)
            accepted = analysis["reliable"]
            final = ({"status": "s21_stage_complete", "next_stage": "spectroscopy",
                      "proposed_updates": {"readout_frequency_hz": analysis["fit"]["resonance_hz"]},
                      "reason": "The numerical S21 checks pass. Propose spectroscopy; full IQ acceptance has not been evaluated."}
                     if accepted else {"status": "stopped", "next_stage": None, "proposed_updates": {},
                                       "reason": "S21 quality is insufficient after the scan budget. Do not advance or claim calibration success."})
            messages.append({"role": "assistant", "content": json.dumps(final)})
            row = {"id": episode_id, "task": "s21_next_action_pilot", "device_id": f"device_{device:03d}",
                   "scenario": scenario, "schema_version": VERSION, "tools": TOOLS, "messages": messages,
                   "label_source": "executed_deterministic_pilot_policy", "training_ready": False}
            buckets[assignment[device]].append(row)
            # Truth remains evaluator-only, absent from all messages and measurement artifacts.
            dump(output / "evaluator_only" / f"{episode_id}.json", {"truth": truth, "trace": trace, "outcome": final})
    for split, rows in buckets.items():
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    manifest = {"schema_version": VERSION, "training_ready": False, "scope": "ideal_calibrated_S21_only",
                "devices": devices, "counts": {key: len(rows) for key, rows in buckets.items()},
                "split_unit": "device; all scan scenarios stay together", "split_seed": 20260904,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(__file__).with_name("s21.py"))},
                "limitations": ["Not full tune-up trajectories", "Not a live Agent adapter",
                                "No independent simulator validation", "No fit uncertainty implementation",
                                "Legacy trainer is incompatible without v2 serialization/masking",
                                "No out-of-family evaluation"]}
    dump(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# S21 Tool-Use Pilot\n\nGenerated simulation artifacts and English tool-use episodes. "
        "**Not training-ready.** See `manifest.json` and `../docs/DATASET_V2.md`. "
        "Keep `evaluator_only/` out of model inputs. The train/validation/test splits are device-grouped "
        "format tests, not a full calibration benchmark.\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.devices), indent=2))


if __name__ == "__main__":
    main()
