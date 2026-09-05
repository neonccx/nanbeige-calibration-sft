# H100 deployment notes

This project stores no server credentials, hostnames or private network addresses. Substitute your
own absolute paths and connect with normal system SSH authentication.

Model weights, adapters, evaluation outputs and virtual environments are excluded from Git. On the
research server they live beside this source tree. `deploy/h100-python.sh` is a process-local launcher
for the audited CUDA driver/library combination; it does not modify system libraries or shell startup
files. Revalidate it after any driver change.

The required order is:

1. verify files, package versions and GPU loading;
2. record immutable pre-training metrics;
3. audit the exact tokenizer/chat template and assistant-only loss boundary;
4. run the bounded trainer smoke test and longest-sample memory check;
5. train into a new append-only output directory;
6. freeze the adapter, then run test, OOD, closed-loop and prompt-ablation evaluations.

The completed v2 run used the commands and safeguards in [`TRAINING_V2.md`](TRAINING_V2.md).
Smoke adapters and interrupted runs are preserved as diagnostics but are not release checkpoints.
Scientific failures may produce a nonzero Agent exit status and must not be hidden as software
success. Never activate an adapter automatically merely because a training process exited.
