# Native ZPA2D/XEB-style v3 experiment

## Frozen boundary

Dataset v2, its checkpoints and all `evaluation_v2` evidence remain unchanged. Dataset v3 is an
independent protocol `calibration-step-0.3` corpus. The v2 LoRA is not reused as the v3 candidate.

The model receives controller state and deterministic fit results, never raw arrays or hidden
simulator truth. Targets select the next experiment. ZPA2D fit arithmetic is performed by a tool;
the target applies `sweet_spot_zpa` as `state.z_bias`. XEB is a synthetic single-qubit decay proxy.

## Data integrity result

The local static audit passed:

- 1,839 samples over 24/4/4/8 train/validation/test/OOD devices;
- 1,598 referenced artifacts, all matching their SHA-256 names;
- 155 ZPA2D and 160 XEB native-call targets;
- rectangular ZPA2D axes and I/Q shapes valid;
- no device split overlap and no named simulator-truth field in model-visible rows.

## Ordered GPU job

The ordered pipeline records the tokenizer/template audit and unmodified v3 test/OOD baseline before
creating any adapter, then trains one epoch of BF16 LoRA (rank 32, alpha 64, dropout 0.05; effective
batch 10), evaluates the frozen test/OOD sets and runs identical fresh-seed base/SFT closed loops.
Frozen-context evaluation uses deterministic greedy decoding in batches of four. This batch size was
validated on the H100 with 8k-token contexts and is recorded in every evaluation config and metrics
file. The launcher enables expandable CUDA allocator segments and the evaluator releases inactive
blocks between batches to avoid fragmentation from the model's full-sequence FP32 logits. The
interactive one-request path is unchanged.

```bash
cd /home/caochuangxin/bishe/Nanbeige4.2-3B
nohup bash deploy/h100-python.sh deploy/run_v3_pipeline.py \
  --run-dir runs/v3_20260909_r3 \
  --verification runs/v2_20260904/assistant_loss_verification.json \
  > runs/v3_20260909_r3.launch.log 2>&1 &
```

Every output path is new. `status.json` preserves exact commands, stage return codes, timestamps and
source hashes. A failed scientific closed loop remains recorded rather than converted into success.

## Scientific limits

The flux model uses the leading asymmetric-SQUID transmon approximation and reduced dispersive shift,
not a multilevel solver or calibrated flux-line transfer function. The XEB-shaped decay is not a
multiqubit random-circuit-sampling benchmark and does not replace standard single-qubit RB. No coupler,
hardware backend or instrument call is present.

## Executed RTX 5090 run

The H100 host reported an uncorrectable row-remapping failure and repeated CUDA
`cudaErrorContained` failures during inference. Those failed run directories are retained as
diagnostic evidence and are not model results. The replacement run uses one RTX 5090, a dedicated
venv that reuses server-local PyTorch/CUDA, and the same pinned Transformers/PEFT versions and
frozen dataset. Explicit paths avoid copying or modifying the official base checkpoint:

```bash
cd /home/caochuangxin/bishe/nanbeige-calibration-sft
CUDA_VISIBLE_DEVICES=1 \
QCAL_BISHE=/home/caochuangxin/bishe \
QCAL_PYTHON=/home/caochuangxin/bishe/envs/nanbeige5090/bin/python \
nohup bash deploy/h100-python.sh deploy/run_v3_pipeline.py \
  --model /home/caochuangxin/bishe/models/Nanbeige4.2-3B \
  --agent /home/caochuangxin/bishe/quantum-calibration-agent \
  --run-dir runs/v3_20260909_5090 \
  --verification runs/prerequisites/assistant_loss_verification_v2.json \
  > runs/v3_20260909_5090.launch.log 2>&1 &
```

The run completed one epoch and 103 optimizer steps with BF16 LoRA rank 32, alpha 64, dropout 0.05,
effective batch 10 and assistant-only sparse projection. Training loss was 0.1177623 and recorded
training runtime was 5,678.3 seconds.

The shared server was under contention: the first ordered process was terminated during SFT OOD
evaluation and a later attempt encountered GPU memory pressure while unrelated jobs occupied all four
GPUs. Completed training and prior evaluations were left immutable. The recovery script validated the
configuration, dataset hash and exact 100-record prediction-ID prefix before appending the remaining
350 OOD predictions. It then ran controller scoring and both fresh-seed closed loops. The recovery
status is `completed_with_failures` because failed scientific episodes deliberately return nonzero.

| Split / policy | Valid | Next tool | Arguments | Controller-executable | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| test base (182) | 0.9176 | 0.5604 | 0.1044 | not scored | 3.94 s |
| test SFT (182) | 1.0000 | 0.9780 | 0.9341 | 0.9890 | 3.52 s |
| OOD base (450) | 0.9311 | 0.6000 | 0.0889 | not scored | 3.68 s |
| OOD SFT (450) | 1.0000 | 0.9089 | 0.8689 | 0.9222 | 3.48 s |

The base policy accepted 0/3 fresh-seed simulated episodes. SFT accepted 2/3 in 13 and 12
experiments; both ended with two independent held-out IQ passes. The third ended in a recorded policy
error after three experiments. The versioned adapter archive has SHA-256
`75c2c83a9ff6797ec5803dede08839bcb77138d9200f631e1c65470cff27bb69`.
