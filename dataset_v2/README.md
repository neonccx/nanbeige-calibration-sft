# Executed calibration dataset v2

304 trajectories from 80 grouped simulated devices; 2,705 next-action targets.
Use `train.jsonl` (1,670 samples) and `validation.jsonl` (280) for SFT. Freeze `test.jsonl`
(277) and `ood.jsonl` (478) for evaluation. The split is by device, never by individual rows.

`artifacts/` contains 2,321 immutable raw acquisitions addressed by SHA-256. `trajectories/`
contains full executed conversations and terminal outcomes; `audit/` preserves controller events.
`evaluator_only/` holds latent parameters and scenario seeds; it is never loaded by SFT.

Targets use the Nanbeige native JSON `calibration.step` function-call format. The runtime executes
the named experiment and registered deterministic analysis as one bounded workflow operation.
Each SFT row contains the same bounded controller context used online and one assistant target.
Only that final assistant span receives loss. This is not training the model to reproduce raw IQ
arrays, invent numerical fits, or execute arbitrary generated Python.

Successful, retrying and escalated trajectories are retained. The teacher is deterministic and
observable-only; matching it measures imitation, not proof of optimal calibration. Drift,
quasi-static dephasing and extreme-noise profiles are held out as OOD stress tests.

Generation and all hashes are recorded in `manifest.json`. `source_sha256` identifies the exact
Agent source snapshot used to generate this immutable dataset; later Agent releases may have
different hashes as the CLI, evaluation harness and prompt profiles evolve. `split_sha256` remains
the byte-level integrity check for the released JSONL files. The model-specific tokenizer audit
passed all rows: maximum 7,450 tokens; target loss spans 53–115 tokens. Use max length 8,192,
no truncation, and a recorded pre-training baseline before starting v2 LoRA.

This is a research candidate, not a hardware-validated dataset. The fuller aliasing,
stale-calibration and reviewed offline tool-development curricula remain separate extensions.
