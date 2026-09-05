"""Record deployment integrity and optional real GPU model inference before training."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refusing to replace a previous record")
    import torch
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = args.project / "models/Nanbeige4.2-3B"
    index = json.loads((root / "model.safetensors.index.json").read_text())
    manifest = json.loads((args.project / "deploy/model_source_sha256.json").read_text())
    report = {"environment": {"python": platform.python_version(), "hostname": platform.node(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "peft", "accelerate", "datasets", "tokenizers")},
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
              "model_files": {}, "datasets": {}}
    for name in sorted(set(index["weight_map"].values())):
        path = root / name
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_size))
        expected = 8 + header_size + max(item["data_offsets"][1] for key, item in header.items() if key != "__metadata__")
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete shard: {name}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            missing = {key for key, shard in index["weight_map"].items() if shard == name} - set(handle.keys())
        if missing:
            raise ValueError(f"Missing tensors: {missing}")
        report["model_files"][name] = {"bytes": expected, "sha256": sha256(path)}
        if report["model_files"][name]["sha256"] != manifest["files"][name]:
            raise ValueError(f"Source/destination SHA-256 mismatch: {name}")
        print("verified", name, flush=True)
    for split in ("train", "validation", "test"):
        path = args.project / "dataset" / f"{split}_recommended.jsonl"
        with path.open() as handle:
            count = sum(1 for _ in handle)
        report["datasets"][split] = {"rows": count, "sha256": sha256(path)}
    if args.load_model:
        if torch.cuda.mem_get_info()[0] < 32 * 1024**3:
            raise RuntimeError("Less than 32 GiB GPU memory free; defer model probe")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
        loaded = time.time()
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with only: hello."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
        batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**batch, do_sample=False, max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        report["generation_probe"] = {"prompt": prompt,
            "output": tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True),
            "load_seconds": loaded - start, "generate_seconds": time.time() - loaded,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "scope": "readiness check, NOT a calibration performance benchmark"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
