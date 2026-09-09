#!/usr/bin/env python3
"""Resume v3 post-training evaluation without modifying completed evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def count_lines(path):
    with path.open() as handle:
        return sum(1 for _ in handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    root = args.run_dir.resolve()
    model = args.model.resolve()
    agent = args.agent.resolve()
    dataset = project / "dataset_v3"
    adapter = root / "training" / "final_adapter"
    original_status = root / "status.json"

    required = {
        "original status": original_status,
        "base model config": model / "config.json",
        "agent package": agent / "src" / "qmagent",
        "dataset manifest": dataset / "manifest.json",
        "adapter config": adapter / "adapter_config.json",
        "adapter weights": adapter / "adapter_model.safetensors",
        "SFT test metrics": root / "sft_test" / "metrics.json",
        "SFT test predictions": root / "sft_test" / "predictions.jsonl",
        "test controller score": root / "sft_test" / "controller_score.json",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        raise ValueError("Cannot resume; missing: " + ", ".join(missing))
    if count_lines(required["SFT test predictions"]) != 182:
        raise ValueError("Cannot resume; SFT test predictions are incomplete")

    original = json.loads(original_status.read_text())
    if original.get("status") != "failed" or original.get("current_stage") != "sft_ood":
        raise ValueError("Resume is only valid after the recorded sft_ood interruption")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ood_output = root / "sft_ood"
    resume_ood = ((ood_output / "config.json").is_file()
                  and (ood_output / "predictions.jsonl").is_file()
                  and not (ood_output / "metrics.json").exists())
    preserved = []
    # Every retry is auditable. Preserve partial output and the preceding
    # recovery state/log instead of truncating or silently reusing either.
    retry_artifacts = [root / "recovery_status.json", root / "recovery_sft_ood.log"]
    if not resume_ood:
        retry_artifacts.extend((root / "sft_ood", root / "sft_ood.log"))
    for path in retry_artifacts:
        if path.exists():
            destination = root / f"{path.name}.interrupted_{stamp}"
            shutil.move(path, destination)
            preserved.append(str(destination))

    state = {
        "pid": os.getpid(),
        "status": "running",
        "started": utc_now(),
        "run_dir": str(root),
        "original_status_sha256": hashlib.sha256(original_status.read_bytes()).hexdigest(),
        "resume_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "preserved_interrupted_artifacts": preserved,
        "resumed_ood_prediction_prefix": (
            count_lines(ood_output / "predictions.jsonl") if resume_ood else 0),
        "scope": "Resume post-training evaluation only; completed training and test evidence are immutable",
        "stages": [],
    }
    status_path = root / "recovery_status.json"
    launcher = ["bash", str(project / "deploy" / "h100-python.sh")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTHONPATH=str(agent / "src"))

    def save():
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    def run(name, command, cwd, scientific_failure=False):
        item = {"stage": name, "command": list(map(str, command)), "status": "running"}
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        log_path = root / ("recovery_" + name + ".log")
        with log_path.open("x") as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {log_path.name}")

    try:
        out = ood_output
        evaluation_command = launcher + [agent / "scripts" / "evaluate_policy_v2.py",
            "--model", model, "--adapter", adapter, "--trust-remote-code",
            "--test-file", dataset / "ood.jsonl", "--output", out,
            "--batch-size", "4"]
        if resume_ood:
            evaluation_command.append("--resume")
        run("sft_ood", evaluation_command, agent)
        run("controller_ood", launcher + [agent / "scripts" / "score_policy_predictions.py",
            "--test-file", dataset / "ood.jsonl", "--predictions", out / "predictions.jsonl",
            "--output", out / "controller_score.json"], agent)
        for arm in ("base", "sft"):
            command = launcher + ["-m", "qmagent.cli", "run", "--policy", "hf",
                "--decode-backend", "hf", "--model", model, "--trust-remote-code",
                "--backend", "physical", "--seed", "2026090650", "--episodes", "3",
                "--max-steps", "40", "--max-tool-calls", "10", "--request-timeout", "300",
                "--output-dir", root / ("closed_loop_" + arm)]
            if arm == "sft":
                command += ["--adapter", adapter]
            run("closed_loop_" + arm, command, agent, scientific_failure=True)
        state["status"] = "completed_with_failures" if any(
            item["status"] == "failed" for item in state["stages"]) else "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished"] = utc_now()
        save()


if __name__ == "__main__":
    main()
