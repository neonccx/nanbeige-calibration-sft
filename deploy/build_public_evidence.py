"""Build a path-free, machine-readable summary from immutable evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TERMINAL = {"completed", "completed_with_failures"}
METRIC_KEYS = (
    "sample_count",
    "valid_rate",
    "next_tool_correct_rate",
    "arguments_correct_rate",
    "mean_seconds",
)


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_arm(directory: Path):
    metrics_path = directory / "metrics.json"
    controller_path = directory / "controller_score.json"
    metrics = load(metrics_path)
    controller = load(controller_path)
    return {
        **{key: metrics[key] for key in METRIC_KEYS},
        "controller_executable_rate": controller["controller_executable_rate"],
        "dataset_sha256": metrics["dataset_sha256"],
        "artifact_sha256": {
            "metrics.json": sha256(metrics_path),
            "controller_score.json": sha256(controller_path),
            "predictions.jsonl": sha256(directory / "predictions.jsonl"),
        },
    }


def comparison(path: Path):
    report = load(path)
    metadata = report.get("comparison")
    if metadata and metadata.get("dimension") == "weights":
        metadata = {
            "dimension": "weights",
            "baseline_adapter_present": metadata.get("baseline_adapter") is not None,
            "candidate_adapter_present": metadata.get("candidate_adapter") is not None,
        }
    return {
        "sample_count": report["sample_count"],
        "scope": report["scope"],
        "comparison": metadata,
        "metrics": {
            key: {field: value[field] for field in ("baseline", "adapted", "delta_percentage_points")}
            for key, value in report["metrics"].items()
        },
        "artifact_sha256": sha256(path),
    }


def closed_loop(path: Path):
    report = load(path)
    episodes = []
    for row in report["episode_results"]:
        gate = row.get("final_iq_gate")
        episodes.append({
            "seed": row["seed"],
            "status": row["status"],
            "reason": row["reason"],
            "experiment_count": row["experiment_count"],
            "tool_counts": row["tool_counts"],
            "consecutive_iq_passes": row["consecutive_iq_passes"],
            "final_iq_metrics": gate["metrics"] if gate else None,
            "final_iq_passed": gate["passed"] if gate else None,
        })
    return {
        "episodes": report["episodes"],
        "statuses": report["statuses"],
        "acceptance_rate": report.get("acceptance_rate"),
        "mean_experiments_success": report["mean_experiments_success"],
        "episode_results": episodes,
        "artifact_sha256": sha256(path),
    }


def require_stages(status, names):
    succeeded = {row["name"] for row in status["stages"] if row.get("returncode") == 0}
    missing = set(names) - succeeded
    if missing:
        raise ValueError(f"Required stages did not complete: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite public evidence")

    main_status = load(args.main_root / "status.json")
    if main_status.get("status") not in TERMINAL:
        raise ValueError("Main evaluation is not terminal")
    required = [f"{split}_{arm}" for split in ("test", "ood") for arm in ("base", "sft")]
    required += [f"{split}_{arm}_controller" for split in ("test", "ood") for arm in ("base", "sft")]
    required += ["test_comparison", "ood_comparison"]
    require_stages(main_status, required)

    fit_status = load(args.fit_root / "status.json")
    if fit_status.get("status") != "completed" or fit_status.get("returncode") != 0:
        raise ValueError("Fit-tool evaluation is not successfully terminal")

    report = {
        "schema": "qcal-public-evidence-0.1",
        "scope": "Synthetic frozen-context and reduced-model evidence; not hardware success",
        "full_context": {},
        "closed_loop": {
            "base": closed_loop(args.main_root / "fresh_base" / "summary.json"),
            "sft": closed_loop(args.main_root / "fresh_sft" / "summary.json"),
            "sft_with_fit_tool": closed_loop(args.fit_root / "closed_loop" / "summary.json"),
        },
        "fit_tool": {
            "known_failure_probe_sha256": sha256(args.fit_root / "probe.json"),
            "token_audit": load(args.fit_root / "token_audit_v2.json"),
        },
        "prompt_ablation": None,
    }
    for split in ("test", "ood"):
        report["full_context"][split] = {
            arm: metric_arm(args.main_root / f"{split}_{arm}") for arm in ("base", "sft")
        }
        report["full_context"][split]["comparison"] = comparison(
            args.main_root / f"{split}_comparison.json"
        )

    if args.prompt_root:
        prompt_status = load(args.prompt_root / "status.json")
        if prompt_status.get("status") not in TERMINAL:
            raise ValueError("Prompt ablation is not terminal")
        prompt_required = [f"{split}_{arm}" for split in ("test", "ood") for arm in ("base", "sft")]
        prompt_required += [f"{split}_{arm}_controller" for split in ("test", "ood") for arm in ("base", "sft")]
        prompt_required += [
            f"{split}_{kind}"
            for split in ("test", "ood")
            for kind in ("base_prompt_comparison", "sft_prompt_comparison", "minimal_training_comparison")
        ]
        require_stages(prompt_status, prompt_required)
        report["prompt_ablation"] = {}
        for split in ("test", "ood"):
            report["prompt_ablation"][split] = {
                arm: metric_arm(args.prompt_root / f"{split}_{arm}") for arm in ("base", "sft")
            }
            for kind in ("base_prompt", "sft_prompt", "minimal_training"):
                report["prompt_ablation"][split][kind + "_comparison"] = comparison(
                    args.prompt_root / f"{split}_{kind}_comparison.json"
                )
        for arm in ("base", "sft"):
            report["prompt_ablation"][f"fresh_{arm}"] = closed_loop(
                args.prompt_root / f"fresh_{arm}" / "summary.json"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
