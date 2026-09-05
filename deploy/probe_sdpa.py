"""Small decode-shaped attention diagnostic; no model weights or training."""

import contextlib
import json
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(2)
torch.manual_seed(17)
report = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
          "cudnn_sdpa_enabled": torch.backends.cuda.cudnn_sdp_enabled(), "cases": []}
for backend in ("default", "no_cudnn", "flash"):
    torch.backends.cuda.enable_cudnn_sdp(backend == "default")
    for length in (4096, 4097):
        q = torch.randn(1, 48, 1, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 48, length, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        torch.cuda.synchronize()
        start = time.perf_counter()
        ctx = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if backend == "flash" else contextlib.nullcontext()
        with ctx, torch.inference_mode():
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        case = {"backend": backend, "kv_length": length, "wall_seconds": time.perf_counter() - start,
                "finite": bool(torch.isfinite(output).all().item())}
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
print(json.dumps(report, indent=2))
