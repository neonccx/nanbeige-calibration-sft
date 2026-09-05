"""Evaluate the frozen v2 adapter on new seeds and complete test/OOD splits."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026090500)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    agent = project.parent/"quantum-calibration-agent"
    model = project/"models/Nanbeige4.2-3B"
    seeds = set(range(args.seed, args.seed+args.episodes))
    known = {json.loads(p.read_text())["seed"] for p in (project/"dataset_v2/evaluator_only").glob("*.json")}
    if not 1 <= args.episodes <= 100 or seeds & known:
        raise ValueError("Invalid episode count or seed overlap with the frozen dataset")
    if not (args.adapter/"adapter_config.json").is_file():
        raise ValueError("Final adapter is missing")
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), status="running", seeds=sorted(seeds), dataset_seed_overlap=[], stages=[])
    launcher = ["bash", str(project/"deploy/h100-python.sh")]
    env = dict(os.environ, PYTHONPATH=str(agent/"src"), PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
    def save():
        temporary = args.output/"status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(args.output/"status.json")
    def run(name, arguments, allowed_failure=False):
        command = launcher+list(map(str, arguments))
        item = dict(name=name, command=command, started=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        with (args.output/(name+".log")).open("x") as log:
            code = subprocess.run(command, cwd=agent, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        item.update(returncode=code, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        if code and not allowed_failure:
            raise RuntimeError(f"Stage {name} failed: {code}")
    try:
        for arm in ("base", "sft"):
            extra = ["--adapter", args.adapter] if arm == "sft" else []
            run("fresh_"+arm, ["-m", "qmagent.cli", "run", "--policy", "hf", "--decode-backend", "hf",
                "--model", model, "--trust-remote-code", "--backend", "physical", "--seed", args.seed,
                "--episodes", args.episodes, "--request-timeout", "300", "--max-steps", "30",
                "--max-tool-calls", "8", "--output-dir", args.output/("fresh_"+arm)]+extra, allowed_failure=True)
        for split in ("test", "ood"):
            for arm in ("base", "sft"):
                name = split+"_"+arm
                directory = args.output/name
                extra = ["--adapter", args.adapter] if arm == "sft" else []
                run(name, [agent/"scripts/evaluate_policy_v2.py", "--model", model, "--trust-remote-code",
                    "--test-file", project/"dataset_v2"/(split+".jsonl"), "--output", directory]+extra)
                run(name+"_controller", [agent/"scripts/score_policy_predictions.py", "--test-file",
                    project/"dataset_v2"/(split+".jsonl"), "--predictions", directory/"predictions.jsonl",
                    "--output", directory/"controller_score.json"])
            run(split+"_comparison", [agent/"scripts/compare_policy_v2.py", "--baseline", args.output/(split+"_base"),
                "--adapted", args.output/(split+"_sft"), "--output", args.output/(split+"_comparison.json")])
        state["status"] = "completed_with_failures" if any(item["returncode"] for item in state["stages"]) else "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
