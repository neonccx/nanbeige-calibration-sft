# Legacy training method (v1.1)

The superseded plan trained Nanbeige4.2-3B on a mixture of Q1-Q9 and bundled policy answers. It
proposed assistant-only BF16 LoRA with no silent truncation, device-grouped splits, a pre-training
baseline, and a separate closed-loop evaluation. Those sound controls were retained.

The old recipe proposed rank 32, alpha 64, dropout 0.05, attention/MLP projection targets, cosine
scheduling and 20,480-token inputs. The installed tokenizer measured a legacy maximum of 19,914
tokens. These values are historical candidates, not the configuration of the completed v2 run.

The active v2 training instead teaches one next native tool call per controller turn. Raw experiment
analysis, fit reliability and numerical extraction belong to deterministic registered tools. See
[`../TRAINING_V2.md`](../TRAINING_V2.md) for the executed recipe and its measured optimization
diagnostics.
