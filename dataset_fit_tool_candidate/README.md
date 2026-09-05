# Fit-tool curriculum candidate

This is a protocol-only reserialization of the frozen v2 trajectories. It does not add synthetic
experiments or move devices between splits. For reliable experiment fits whose numerical updates
can be reproduced exactly by `analysis.calculate_fit_updates`, the assistant target calls
`calibration.step_from_fit`; all other targets remain `calibration.step`.

The intended learning problem is next-action selection. Frequency, amplitude, coherence-time and
reset-delay arithmetic is delegated to a deterministic, versioned analysis tool and is still
checked by the controller before state is committed.

The installed Nanbeige tokenizer audit passed all 2,705 rows. It verified file hashes, equality of
training and runtime prompts, an exact assistant-loss boundary, decodability of every tool target,
and equality of the resolved controller action. The maximum sequence length is 7,724 tokens and
the shortest supervised suffix is 52 tokens; see `token_audit_v2.json`. The earlier
`token_audit.json` is retained as evidence for the pre-audit status manifest.

`training_ready` intentionally remains false until the ordered real-model protocol probe finishes.
If the existing adapter selects the new function reliably and fixes the two known Ramsey arithmetic
failures, no additional training is justified. Otherwise this candidate becomes the next SFT input.

The JSONL metadata still references artifacts in the original `dataset_v2` tree; this directory is
therefore not a standalone dataset bundle.
