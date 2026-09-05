# Graduation-project server layout

- `Nanbeige4.2-3B/`: training code, datasets and evaluation metadata. Model weights remain local.
- `quantum-calibration-agent/`: independent simulation-first Agent runtime.
- `envs/nanbeige445/`: audited compatibility environment that reuses the server CUDA/PyTorch base.

Use the process-local launcher instead of modifying system NVIDIA libraries:

```bash
cd /absolute/path/to/Nanbeige4.2-3B
sh deploy/h100-python.sh -c 'import torch; print(torch.cuda.get_device_name(0))'
```

See `docs/H100_DEPLOYMENT.md` and `docs/TRAINING_V2.md`. A readiness file, log, smoke adapter or
output directory is not evidence that training or evaluation succeeded; require the corresponding
final status and metrics. Preserve interrupted jobs and scientific failures rather than overwriting
them. This repository contains no SSH credentials or private server address.
