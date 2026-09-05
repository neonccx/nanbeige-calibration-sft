# Evaluation protocol and evidence status

## Claims boundary

Training loss, teacher-action agreement, controller executability, simulated closed-loop success
and external plot understanding are different measurements. This project reports them separately.
It does not convert one metric into another or treat a rule-policy plot as evidence that the
language model completed calibration.

## Completed pre-training diagnostic

Before LoRA optimization, the unmodified Nanbeige4.2-3B instruction checkpoint was evaluated on a
frozen 36-example action-stratified subset of the v2 test split with the full workflow instruction.
The preserved metrics are:

| Metric | Base checkpoint |
| --- | ---: |
| native schema validity | 0.9531 |
| next-tool agreement | 0.5957 |
| argument agreement | 0.0975 |
| controller score | 0.4982 |
| mean generation latency | 6.68 s/example |

This subset is an infrastructure and behavior diagnostic, not the final test-set score. Invalid
outputs count as failures. Latency is specific to the recorded server, software stack and decode
configuration and should not be generalized as a model property.

## Ordered frozen evaluation

The final adapter is frozen. The following jobs use new output directories and run in this order so
that they do not contend for GPU memory:

1. full 277-example test comparison, base versus SFT;
2. full 478-example OOD comparison, base versus SFT;
3. deterministic fit-tool verification, including known Ramsey arithmetic failures and three new
   simulator seeds;
4. B0/F0 minimal-instruction prompt ablation on the same frozen test/OOD rows and three fresh
   closed-loop seeds.

Each arm preserves predictions, failures, hashes, timings, controller scores and the exact system
prompt profile. Results will be added only after every ordered process reaches a terminal state and
the artifacts are copied from the H100 host. Until then, no full-split improvement or
generalization claim is made.

## Closed-loop simulation

Fresh-seed episodes use identical simulator seeds, budgets, controller, tool implementations and
decode settings between base and SFT. The model may select only a typed `calibration.step` action.
The controller owns prerequisites, bounds, state commits, experiment budgets and final acceptance.
An episode passes only after two independent held-out IQ acquisitions satisfy all fixed gates.

This is a reduced simulator evaluation. It is not hardware validation and does not authorize direct
instrument control.

## QCalEval

External plot understanding is pinned to NVIDIA QCalEval source commit
`e9e9b9eb8b95f93e0db1c578e699fc212cf0bb84` and dataset revision
`b611794244a251c47fc67eb801b92f05722c369c` (243 rows, CC-BY-4.0). The Agent-side runner preserves
all six questions but locally scores only the deterministic Q2, Q4, Q5 and Q6 components; Q1 and Q3
require the official judge. Therefore its subtotal is explicitly not an official QCalEval overall
score and is never merged with closed-loop acceptance.

The external dataset is not bundled with this repository and was not used for training. At the
current release checkpoint, its server-side snapshot is still unavailable because the pinned files
could not be downloaded reliably. No QCalEval result is claimed.
