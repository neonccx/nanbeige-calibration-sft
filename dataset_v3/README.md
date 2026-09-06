# Dataset v3

Independent simulation-only extension of the frozen v2 dataset. It uses protocol
`calibration-step-0.3`, runtime schema `runtime-0.2` and physics
`reduced-cqed-flux-xeb-0.3`.

| Split | Devices | Examples |
| --- | ---: | ---: |
| train | 24 | 1,032 |
| validation | 4 | 175 |
| test | 4 | 182 |
| OOD | 8 | 450 |

The target distribution includes 155 `sq.s21_zpa2d` and 160 `sq.xeb` actions. Raw observations are
content-addressed under `artifacts/`; model rows contain only public controller context, deterministic
fits, shapes and hashes. Hidden device parameters live only under `evaluator_only/`.

Generation and integrity checks are implemented in the Agent repository:

```bash
PYTHONPATH=src python scripts/build_dataset_v3.py --devices 32 --output /new/path/dataset_v3
PYTHONPATH=src python scripts/audit_dataset_v3.py /new/path/dataset_v3
```

The recorded build passed 1,598 artifact hashes, ZPA2D rectangular-shape checks, device-disjoint
splits and a named-truth leakage scan. This is synthetic data, not hardware measurement. The
single-qubit XEB-style stage is only a decay proxy and is not equivalent to multiqubit XEB.
