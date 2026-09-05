# Native tool-call LoRA experiment

## Objective and boundary

Teach the unmodified Nanbeige4.2-3B instruction checkpoint to select the next calibration action
from public state and deterministic analysis outputs. "Base" in these comparisons means the
unmodified instruction checkpoint, not an unavailable pretraining-only model.

The Agent is software, not another trainable neural network. Its numerical tools, bounds,
quality checks and IQ acceptance are held fixed. Only the model's LoRA adapter is trained.

## Data construction

1. Sample one shared reduced physical device; derive all seven experiments from its parameters.
2. Execute an observation-only teacher through the same controller used online.
3. Store raw acquisitions as immutable hash-addressed artifacts; store fits separately from truth.
4. Keep successful and failed trajectories; include rescan and safe escalation actions.
5. Group every replica of a device into one split; reserve selected regimes as OOD.
6. Export bounded-context next-action samples plus full trajectories for audit.
7. Render with the installed checkpoint's native template using `tools`, `tool_call_format="json"`,
   `enable_thinking=False`, and `preserve_thinking=False`.
8. Require exact prompt-token prefix equality. Mask the entire prompt; train only the final
   assistant tool-call span. Never truncate a target.

Counts: 48 train devices / 8 validation / 8 test / 16 OOD; 1,670 / 280 / 277 / 478 samples.
The longest native sequence is 7,450 tokens. Argument dictionaries are serialized as JSON before
Arrow ingestion so heterogeneous tool parameters cannot acquire spurious null-valued keys.

## Staged experiment

First save an unmodified-model baseline on the frozen 36-case action-stratified test subset.
This is a small diagnostic baseline, not a full test-set result. Save predictions, exact selected IDs,
tokenizer/dataset hashes, timing and failure records under the user's H100 project before any update.

Then run a two-step infrastructure smoke test in a separate output directory. If successful, train
an initial one-epoch BF16 LoRA experiment on the complete training split, rank 32, alpha 64,
dropout 0.05, learning rate 1e-4, gradient clipping 1.0, cosine schedule and 3% warmup.
Use the validation split for diagnostics and any later model selection, not the frozen test subset. Preserve the unmodified
checkpoint and previous adapters. A smoke adapter must never be called a trained calibration policy.

### Audited memory optimization

`--assistant-only-projection` keeps the entire decoder context but applies the vocabulary head only
at supervised causal positions. This preserves the masked mean-cross-entropy objective, rather than
truncating long input or discarding assistant tokens. It is restricted to the audited Nanbeige class
with `pretraining_tp=1`; it is not a generic shortcut for arbitrary models or distributed wrappers.

Two CPU tests compare exact objective/gradients. Real H100 BF16 checks on 1,471- and 7,450-token
samples, with nonzero LoRA B weights and reset dropout RNG, gave equal losses at recorded precision
and relative gradient L2 differences of 1.92% and 1.72% (declared tolerance 5%, not bitwise identity).
Longest-sample peak allocated memory fell from 24,659 to 13,066 MiB in this isolated test.
Timing is a single paired measurement, not a stable throughput benchmark.

The custom Trainer explicitly re-enters `accelerator.autocast()` because direct decoder calls
bypass the automatically wrapped model forward. CUDA BF16 is asserted and logged. The earlier
sparse pipeline without this context was interrupted and preserved; the final corrected smoke
completed with validation loss 0.949893. Current run root: `runs/v2_20260904/sparse_amp_pipeline/`.

The initial full-projection run was intentionally interrupted at step 18/167 before its first
checkpoint; its output directory is preserved and no completed training is claimed for that run.
The revised one-shot pipeline uses micro-batch 2, accumulation 5 (effective batch 10), one epoch,
checkpoint every 25 updates (last three retained), final validation, and the final-epoch adapter.
It does not select checkpoints using the frozen test set and never activates an adapter automatically.
The corrected run completed all 167 updates in 5,801.8 seconds with training loss 0.08077 and
final validation loss 0.003734. These are optimization diagnostics, not calibration acceptance.
`deploy/run_v2_pipeline.py` performs training, the same 36-case evaluation, controller validation,
then three fresh base/SFT simulated closed loops with identical seeds and budgets. Logs and status
remain on the server if SSH disconnects. This is an initial diagnostic experiment, not the full
test/OOD/Ising comparison in the research roadmap.

## Evaluation

Report separately: native schema validity, controller-executable arguments/prerequisites,
next-tool agreement, argument agreement within declared tolerances, measured closed-loop acceptance,
safe escalation, experiment count and latency. Invalid outputs count as failures.

The complete frozen test/OOD comparison and fresh closed loops are reported in
[`EVALUATION.md`](EVALUATION.md). A rule-policy IQ plot is not language-model success. Attractive
plots, loss reduction and teacher agreement do not independently establish full calibration
capability.

The B0 minimal-instruction, B1 workflow-instruction, SFT and Ising comparison arms remain distinct;
this first native baseline uses the frozen domain workflow prompt (B1). Ising's plot-understanding
evaluation requires a separate, pinned vision-language harness, not pretending a text-only metric
is an equivalent comparison.

## Reproduction

Use the project-local H100 Python launcher to resolve the known driver-library mismatch without
changing global driver files. Run `audit_dataset_v2.py`, `evaluate_policy_v2.py`, then `train_lora.py`
with `--baseline-metrics`, `--dataset-audit`, and `--max-length 8192`. Every output directory must be new.
Model files stay on the server; only source, data and small evidence reports are synchronized.
