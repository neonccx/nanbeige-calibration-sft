#!/bin/sh
# Server-local launcher: never change /usr/lib symlinks or global shell config.
set -eu
QCAL_BISHE=${QCAL_BISHE:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}
QCAL_KERNEL_VERSION=$(sed -n 's/.*Kernel Module  *\([^ ]*\).*/\1/p' /proc/driver/nvidia/version)
if [ "$QCAL_KERNEL_VERSION" = "575.57.08" ]; then
    QCAL_CUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.575.57.08
    QCAL_NVML=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.575.57.08
    test -r "$QCAL_CUDA" && test -r "$QCAL_NVML"
    export LD_PRELOAD="$QCAL_CUDA:$QCAL_NVML${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export HF_HOME="$QCAL_BISHE/cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$QCAL_BISHE/envs/nanbeige445/bin/python" "$@"
