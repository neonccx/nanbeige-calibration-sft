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

## H100 job

The ordered pipeline records the tokenizer/template audit and unmodified v3 test/OOD baseline before
creating any adapter, then trains one epoch of BF16 LoRA (rank 32, alpha 64, dropout 0.05; effective
batch 10), evaluates the frozen test/OOD sets and runs identical fresh-seed base/SFT closed loops.
Frozen-context evaluation uses deterministic greedy decoding in batches of five. This batch size was
validated on the H100 with 8k-token contexts and is recorded in every evaluation config and metrics
file. The launcher enables expandable CUDA allocator segments and the evaluator releases inactive
blocks between batches to avoid fragmentation from the model's full-sequence FP32 logits. The
interactive one-request path is unchanged.

```bash
cd /home/caochuangxin/bishe/Nanbeige4.2-3B
nohup bash deploy/h100-python.sh deploy/run_v3_pipeline.py \
  --run-dir runs/v3_20260907_r2 \
  --verification runs/v2_20260904/assistant_loss_verification.json \
  > runs/v3_20260907_r2.launch.log 2>&1 &
```

Every output path is new. `status.json` preserves exact commands, stage return codes, timestamps and
source hashes. A failed scientific closed loop remains recorded rather than converted into success.

## Scientific limits

The flux model uses the leading asymmetric-SQUID transmon approximation and reduced dispersive shift,
not a multilevel solver or calibrated flux-line transfer function. The XEB-shaped decay is not a
multiqubit random-circuit-sampling benchmark and does not replace standard single-qubit RB. No coupler,
hardware backend or instrument call is present.
