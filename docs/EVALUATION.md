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

## Frozen full-split results

The final adapter was frozen before these comparisons. Each arm used the same rows, native tool
schema, full workflow instruction and reference HF decoder. Invalid outputs count as failures.

| Split / policy | Valid | Next tool | Arguments | Controller-executable | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| test base (277) | 0.9531 | 0.5957 | 0.0975 | 0.4982 | 6.68 s |
| test SFT (277) | 1.0000 | 1.0000 | 0.9495 | 0.9819 | 7.76 s |
| OOD base (478) | 0.9163 | 0.6632 | 0.0900 | 0.5230 | 7.72 s |
| OOD SFT (478) | 1.0000 | 0.9289 | 0.8996 | 0.9226 | 16.20 s |

These are frozen-context imitation and executability metrics, not closed-loop or hardware success.
The OOD latency increase is reported as observed and has not yet been attributed to one cause.

## Closed-loop and fit-tool verification

On three fresh simulator seeds, the original base policy accepted 0/3 episodes. The original SFT
policy accepted 2/3 in nine experiments each; its remaining episode was safely rejected after it
requested T2 Echo before the controller had accepted T1. These nonzero process return codes are
scientific outcomes, not missing artifacts.

The registered `calibration.step_from_fit` path was then tested on the same three seeds. Two known
Ramsey arithmetic failures produced controller-executable actions, the complete 2,705-row optional
curriculum passed tokenizer and target-boundary audit, and all three fresh closed loops were
accepted in nine experiments. Every accepted episode ended with two independent held-out IQ passes.
This small 3/3 result demonstrates the intended tool boundary; it is not a statistical hardware
success estimate.

## Remaining prompt ablation

The B0/F0 minimal-instruction prompt ablation is still running in the ordered H100 queue. It keeps
the checkpoint, adapter, rows, native tools, controller, simulator, seeds and decoding fixed while
changing only the system instruction. Its result will be added only after the process reaches a
terminal state and the artifacts are copied from the server.

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
