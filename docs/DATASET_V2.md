# Dataset v2: Learn the Next Action

Status: a seven-stage executed candidate now exists in `dataset_v2/`; legacy `dataset/` and
the S21-only pilot are preserved. All 2,705 native target loss boundaries, 2,321 artifact hashes,
device split isolation and 14 reduced-dynamics checks passed. See [current training method](TRAINING_V2.md).

### Implemented protocol decision

The first release uses one model-visible atomic function, `calibration.step`. Its arguments select
the experiment, bounded updates and rescan parameters; the controller then calls acquisition and its
registered observation-only numerical analysis. The two operations have distinct trace records and
analysis is also callable independently in Python via `analyze`/`analyze_file`. They are not falsely
represented as two separate model decisions. This deliberately keeps tool scheduling compatible with
the current controller while teaching the requested next-experiment decision. Model-directed selection
among alternative numerical fit algorithms is a later extension.

Each training row uses the same bounded state/recent-history view as online inference plus one native
assistant function call. Full executed trajectories retain canonical assistant/tool roles for audit.
The model receives artifact references and exact structured metrics, not raw arrays. The sections below
preserve the broader design specification and S21 pilot history; where they propose separate model-visible
acquisition/analysis calls or a future trainer update, the atomic protocol above is the implemented version.

## Data unit

One episode belongs to one simulated device, with a stable latent parameter set and explicit observation noise. It contains English system/user messages, assistant tool selections with arguments and short evidence-based rationales, actual tool responses, and an observed termination reason. Keep raw arrays in immutable artifacts referenced by ID, path and SHA-256. Do not paste hundreds of numeric rows into a prompt.

The conceptual flow is `request -> measurement tool -> artifact -> analysis tool -> structured quality/result -> next decision`, repeated until independent acceptance or an explicit stop. Measurement and analysis are distinct tool calls. Tool responses are genuine recorded outputs, not text invented to justify a target answer.

## Tool contract

- Versioned names and JSON schema; explicit units and finite values.
- Raw artifact + acquisition metadata from measurement.
- Fit values, uncertainty/diagnostics, reliability flags and provenance from analysis.
- Explicit bounded rescan/update arguments from the model; no vague “increase the range.”
- Controller-owned parameter commits, prerequisites, budget and final acceptance.
- Missing tool: stop/request a reviewed tool implementation, or use a separately sandboxed offline development task. Never run unrestricted generated code on an instrument host.

The pilot exposes `experiment.s21` and `analysis.fit_s21`. These are an **experimental data protocol**, not currently registered live Agent tools. Its final assistant answer is a proposed next stage or a bounded stop, not a fabricated execution of spectroscopy or full calibration success.

## Supervision and model compatibility

Store native structured `tool_calls` and `tool` messages in the canonical dataset. A model-specific serializer must pass the tool schema and preserve call/response associations using the pinned tokenizer's supported convention. Do not relabel tool output as user text merely to make a template run.

The implemented `train_lora.py` loads each prefix-to-next-assistant example together with its native
tool schema, renders it through the pinned Nanbeige chat template and verifies exact prompt-prefix
token equality. It masks the complete prompt and applies loss only to the final assistant tool-call
span. Rows with malformed calls, an empty target, a missing schema or any required truncation are
rejected rather than repaired silently. The same serializer is used by offline evaluation.

Never train loss on tool results, controller events or hidden truth. Prefixes remain in the same
split as their parent episode, and no synthetic hidden-reasoning text is added. The optional sparse
assistant-token vocabulary projection preserves the full decoder context; its objective and LoRA
gradients were checked against full projection on CPU and on the target H100 as described in
[`TRAINING_V2.md`](TRAINING_V2.md).

## Splits and anti-leakage

Group by physical device family and device ID **before** expanding trajectories or prefixes. All noise replicas, retries and scan variants of a device stay together. Reserve unseen parameter/noise/model families for out-of-distribution evaluation. A random row split is forbidden.

Keep truth separately for an evaluator; omit it, future outcomes, reference actions and fitted-on-test thresholds from model-visible messages. Report action validity and final outcomes alongside next-action similarity: more than one action can be defensible.

## Curriculum and coverage

Start with correct tool selection and simple progression; add edge hits, insufficient span, ambiguous peaks, low contrast, failed fits, unit/argument errors, stale calibration, drift, budget exhaustion and safe stopping. Train long recovery trajectories only after the corresponding tools and simulator are validated.

The pilot covers ideal calibrated S21 normal/edge/retry decisions on a small number of devices. It deliberately does **not** claim noise-family diversity, multi-peak behavior, live code generation, independent simulation validation, or final IQ calibration.

Pilot priors are project-selected, not extracted from a paper: resonance uniform in 6.5 GHz +/- 2 MHz; loaded Q uniform from 3,000 to 6,000; notch depth uniform from 0.35 to 0.70; independent Gaussian receiver noise with standard deviation 0.002 per quadrature in normalized S21 units. The response assumes an already calibrated measurement chain. Initial scans are fixed public settings, not centered on hidden device truth. Device sampling seed is `70000 + device_index`; the source records separate acquisition seeds. Scenario labels and evaluator truth must not be passed as model inputs.

## Reproduce the pilot

Use an environment containing NumPy and SciPy. From this project:

```bash
PYTHONPATH=src python -m calibration_v2.build_pilot --output dataset_v2_pilot
PYTHONPATH=src python -m unittest discover -s tests_v2 -v
```

The builder refuses an existing output directory, saves raw measurements separately, hashes artifacts, executes analysis on reloaded observations, groups device replicas, and emits a manifest with `training_ready: false`. A generated split is a **format test**, not a new benchmark score or permission to start SFT.

See [Training v2](TRAINING_V2.md) for the release gates and evaluation matrix. The controlled
prompt comparison is documented in the separate Agent repository's
[`PROMPT_ABLATION.md`](https://github.com/neonccx/quantum-calibration-agent/blob/main/docs/PROMPT_ABLATION.md).
