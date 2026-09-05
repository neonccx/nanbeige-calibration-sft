"""One-shot offline H100 training/evaluation job; no daemon or hardware access."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    agent = project.parent / "quantum-calibration-agent"
    model = project / "models/Nanbeige4.2-3B"
    prior = project / "runs/v2_20260904"
    verification = json.loads(args.verification.read_text())
    if not verification.get("all_passed") or len(verification.get("cases", [])) < 2:
        raise ValueError("Real-model sparse-loss verification has not passed")
    verified_model = Path(verification["model"])
    if not verified_model.is_absolute():
        verified_model = project / verified_model
    if verified_model.resolve() != model.resolve():
        raise ValueError("Verification uses a different checkpoint")
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), status="running", stages=[], started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 verification_sha256=hashlib.sha256(args.verification.read_bytes()).hexdigest(),
                 model=str(model), source_sha256={str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (project / "src/train_lora.py", project / "src/assistant_loss.py", Path(__file__).resolve())},
                 scope="Offline simulated research experiment; never activates the adapter")
    launcher = ["bash", str(project / "deploy/h100-python.sh")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    def save():
        temp = root / "status.json.tmp"
        temp.write_text(json.dumps(state, indent=2)+"\n")
        temp.replace(root / "status.json")
    def run(name, command, cwd, scientific_failure=False):
        item = dict(stage=name, command=list(map(str, command)), status="running")
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        with (root / (name+".log")).open("x") as log:
            result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode, status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect its log")
    train = root / "training"
    evaluation = root / "evaluation_36"
    try:
        run("training", launcher + [project / "src/train_lora.py", "--model", model,
            "--train-file", project / "dataset_v2/train.jsonl", "--validation-file", project / "dataset_v2/validation.jsonl",
            "--output-dir", train, "--baseline-metrics", prior / "baseline_native_36_retry/metrics.json",
            "--dataset-audit", prior / "tokenizer_audit.json", "--max-length", "8192", "--epochs", "1",
            "--micro-batch", "2", "--gradient-accumulation", "5", "--logging-steps", "5",
            "--eval-steps", "167", "--save-steps", "25", "--assistant-only-projection", "--no-load-best-model"], project)
        adapter = train / "final_adapter"
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError("Training did not produce a final adapter")
        run("evaluation", launcher + [agent / "scripts/evaluate_policy_v2.py", "--model", model,
            "--adapter", adapter, "--trust-remote-code", "--test-file", project / "dataset_v2/test.jsonl",
            "--limit", "36", "--output", evaluation], agent)
        run("controller_score", launcher + [agent / "scripts/score_policy_predictions.py",
            "--test-file", project / "dataset_v2/test.jsonl", "--predictions", evaluation / "predictions.jsonl",
            "--output", evaluation / "controller_score.json"], agent)
        # Same fresh seeds and budgets for base and adapted policies. No rule fallback.
        for arm in ("base", "sft"):
            command = launcher + ["-m", "qmagent.cli", "run", "--policy", "hf", "--decode-backend", "hf",
                "--model", model, "--trust-remote-code", "--backend", "physical", "--seed", "2026090456",
                "--episodes", "3", "--max-steps", "30", "--max-tool-calls", "8",
                "--request-timeout", "300", "--output-dir", root / ("closed_loop_"+arm)]
            if arm == "sft":
                command += ["--adapter", adapter]
            env["PYTHONPATH"] = str(agent / "src")
            run("closed_loop_"+arm, command, agent, scientific_failure=True)
        state["status"] = "completed_with_failures" if any(x["status"] == "failed" for x in state["stages"]) else "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
