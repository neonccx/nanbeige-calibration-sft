# Legacy dataset construction (v1.1)

This document preserves the design of the superseded question-answer dataset. The active experiment
uses `dataset_v2` and the native next-action contract instead.

The v1.1 workflow contained S21, spectroscopy, PiAmp, Ramsey, T1, echo and final IQ readout. It
generated 1,352 physical rounds over 122 virtual qubits, then expanded each round into Q1-Q8 prompts
and each trajectory into Q9. The recommended deduplicated split contained 5,001/591/604 examples.

IQ acceptance used a linear discriminator between the two fitted centroids and recorded projected
separation, SNR, F0, F1, visibility and assignment fidelity. The synthetic baseline thresholds were
SNR >= 2.5, visibility >= 0.80, F0/F1 >= 0.88 and assignment fidelity >= 0.90. These thresholds were
never presented as hardware standards.

Complete qubit trajectories were kept within one split. Hidden device truth was reserved for audit,
not inserted into user prompts. Failed trajectories were retained so that the model could learn to
repeat, rescan or escalate rather than declare success from any two visible clusters.

The v1.1 `POLICY` answer bundled diagnosis, extracted metrics, parameter updates, next tool and final
IQ status. This differs from the active v2 contract, where deterministic tools calculate metrics and
the model returns one native bounded action. Old v1.1 adapters are therefore not runtime-compatible.
