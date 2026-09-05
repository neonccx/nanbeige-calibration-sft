# Nanbeige4.2-3B Calibration SFT

Training, data and evaluation project for adapting
[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) to choose the next bounded action in
a simulation-first single-qubit calibration loop. The experiment-executing runtime lives in the
separate [quantum-calibration-agent](https://github.com/neonccx/quantum-calibration-agent) project.

> Research status, 2026-09-05: the one-epoch LoRA run completed. Frozen full test/OOD, fit-tool and
> prompt-ablation evaluations are still running in an ordered H100 queue. No final generalization
> claim is made until those artifacts are complete and copied into the release evidence.

## What the model learns

The model receives only public controller state and registered analysis outputs. It returns exactly
one native `calibration.step` call containing the next experiment, bounded parameter updates or scan
overrides, and a short English reason. It does not fit raw arrays, execute Python/shell commands,
write directly to hardware, or decide final acceptance by itself.

The seven-stage workflow is S21, spectroscopy, PiAmp, Ramsey, T1, echo and IQraw. A deterministic
controller enforces prerequisites, units, parameter bounds, experiment budgets and two independent
held-out IQ passes.

## Active project map

| Path | Purpose |
| --- | --- |
| `dataset_v2/` | 2,705 executed next-action examples and immutable artifacts |
| `dataset_fit_tool_candidate/` | Audited optional curriculum for explicit deterministic fit arithmetic |
| `dataset_v2_pilot/` | Preserved S21-only protocol pilot, not used for final SFT |
| `src/train_lora.py` | Assistant-only BF16 LoRA trainer |
| `src/assistant_loss.py` | Sparse assistant-token objective |
| `src/calibration_v2/` | Shared v2 data and protocol utilities |
| `deploy/` | Reproducible H100 launch, smoke and ordered evaluation scripts |
| `docs/TRAINING_V2.md` | Executed recipe, evidence and limitations |
| `docs/DATASET_V2.md` | Dataset contract and leakage boundary |
| `docs/EVALUATION.md` | Frozen metrics, queued comparisons and claim boundaries |

Base-model weights, virtual environments, checkpoints and raw server logs are deliberately excluded
from Git. The final LoRA adapter should be distributed as a versioned release artifact with its
SHA-256, base-model identifier and exact training/evaluation metadata—not committed to repository
history.

## Dataset

| Split | Devices | Trajectories | Decision examples |
| --- | ---: | ---: | ---: |
| train | 48 | 192 | 1,670 |
| validation | 8 | 32 | 280 |
| test | 8 | 32 | 277 |
| OOD | 16 | 48 | 478 |

All rows are English. Device groups do not cross splits. Raw artifacts are content-addressed;
simulator truth remains in `evaluator_only/` and is never a training input. The environment is a
reduced two-level RWA/Markov/linear-cavity simulator rather than hardware data or a full digital
twin. See [the dataset card](DATA_CARD.md).

## Reproduce the validation gates

Create a compatible environment and install the documented H100 requirements. Download the base
checkpoint separately; no script in this repository downloads weights implicitly.

```bash
python -m pip install -r requirements-h100.txt

python src/verify_assistant_loss.py \
  --model /absolute/path/Nanbeige4.2-3B \
  --data dataset_v2/train.jsonl \
  --output /absolute/new-output/token_audit.json
```

The release gate requires exact tokenizer/chat-template compatibility, no silent truncation, a valid
assistant-only loss boundary for every row, a recorded pre-training baseline, a two-update trainer
smoke test and the longest-sample memory check. Use a new output path for every run.

## Train

The completed research run used one epoch, BF16, LoRA rank 32/alpha 64/dropout 0.05, micro-batch 2,
gradient accumulation 5 and the sparse assistant-token projection. Consult
[`docs/TRAINING_V2.md`](docs/TRAINING_V2.md) for the full command, package versions and measured
optimization diagnostics. Validation loss is not closed-loop calibration success.

## Evaluate

Evaluation has three distinct tracks:

1. frozen-context schema, next-tool, argument and controller-executability metrics on test and OOD;
2. closed-loop simulated device acceptance, experiment count and failure modes;
3. external visual plot understanding through the separate pinned QCalEval protocol.

The B0/B1/F0/F1 ablation changes only the calibration system instruction (`minimal` versus `skill`)
while keeping weights, native tools, public context, controller, simulator, seeds and decoding fixed.
QCalEval plot scores must never be merged with closed-loop IQ acceptance.

## Safety and scope

- Simulation only; no instrument adapter is enabled.
- Synthetic thresholds are research gates, not hardware specifications.
- Real-device work requires typed adapters, unit checks, rollback and human approval.
- Failed and interrupted runs remain evidence and must not be overwritten or reported as successes.
- The underlying Nanbeige checkpoint is Apache-2.0 according to its official model card; preserve its
  license and attribution when distributing an adapter or merged checkpoint.

## Documentation

- [Dataset card](DATA_CARD.md)
- [Dataset v2 specification](docs/DATASET_V2.md)
- [Training v2](docs/TRAINING_V2.md)
- [Evaluation protocol](docs/EVALUATION.md)
- [H100 deployment](docs/H100_DEPLOYMENT.md)
- [Legacy v1.1 notes](docs/archive/DATASET_V1_1.md)
