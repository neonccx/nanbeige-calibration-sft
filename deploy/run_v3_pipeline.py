#!/usr/bin/env python3
"""Audited H100 baseline, LoRA training and v3 test/OOD/closed-loop evaluation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    agent = project.parent/"quantum-calibration-agent"
    model = project/"models/Nanbeige4.2-3B"
    dataset = project/"dataset_v3"
    verification = json.loads(args.verification.read_text())
    if not verification.get("all_passed") or len(verification.get("cases", [])) < 2:
        raise ValueError("Audited real-model sparse-loss verification is required")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = {"pid": os.getpid(), "status": "running", "stages": [],
             "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "model": str(model), "dataset": str(dataset),
             "verification_sha256": hashlib.sha256(args.verification.read_bytes()).hexdigest(),
             "scope": "Synthetic single-qubit v3 research; no hardware or coupler access",
             "source_sha256": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (project/"src/train_lora.py", project/"src/assistant_loss.py", Path(__file__).resolve())}}
    launcher = ["bash", str(project/"deploy/h100-python.sh")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(agent/"src"))

    def save():
        temporary = root/"status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(root/"status.json")

    def run(name, command, cwd, scientific_failure=False):
        item = {"stage": name, "command": list(map(str, command)), "status": "running"}
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        with (root/(name+".log")).open("x") as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {name}.log")

    audit = root/"tokenizer_audit.json"
    adapter = root/"training"/"final_adapter"
    try:
        run("tokenizer_audit", launcher+[agent/"scripts/audit_dataset_v2.py", "--dataset", dataset,
            "--model", model, "--output", audit], agent)
        for split in ("test", "ood"):
            run("baseline_"+split, launcher+[agent/"scripts/evaluate_policy_v2.py", "--model", model,
                "--trust-remote-code", "--test-file", dataset/(split+".jsonl"),
                "--output", root/("baseline_"+split), "--batch-size", "6"], agent)
        run("training", launcher+[project/"src/train_lora.py", "--model", model,
            "--train-file", dataset/"train.jsonl", "--validation-file", dataset/"validation.jsonl",
            "--output-dir", root/"training", "--baseline-metrics", root/"baseline_test"/"metrics.json",
            "--dataset-audit", audit, "--max-length", "20480", "--epochs", "1",
            "--micro-batch", "2", "--gradient-accumulation", "5", "--logging-steps", "5",
            "--eval-steps", "104", "--save-steps", "25", "--assistant-only-projection",
            "--no-load-best-model"], project)
        if not (adapter/"adapter_config.json").is_file():
            raise RuntimeError("Training did not produce final_adapter")
        for split in ("test", "ood"):
            out = root/("sft_"+split)
            run("sft_"+split, launcher+[agent/"scripts/evaluate_policy_v2.py", "--model", model,
                "--adapter", adapter, "--trust-remote-code", "--test-file", dataset/(split+".jsonl"),
                "--output", out, "--batch-size", "6"], agent)
            run("controller_"+split, launcher+[agent/"scripts/score_policy_predictions.py",
                "--test-file", dataset/(split+".jsonl"), "--predictions", out/"predictions.jsonl",
                "--output", out/"controller_score.json"], agent)
        for arm in ("base", "sft"):
            command = launcher+["-m", "qmagent.cli", "run", "--policy", "hf", "--decode-backend", "hf",
                "--model", model, "--trust-remote-code", "--backend", "physical", "--seed", "2026090650",
                "--episodes", "3", "--max-steps", "40", "--max-tool-calls", "10",
                "--request-timeout", "300", "--output-dir", root/("closed_loop_"+arm)]
            if arm == "sft":
                command += ["--adapter", adapter]
            run("closed_loop_"+arm, command, agent, scientific_failure=True)
        state["status"] = "completed_with_failures" if any(
            item["status"] == "failed" for item in state["stages"]) else "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
