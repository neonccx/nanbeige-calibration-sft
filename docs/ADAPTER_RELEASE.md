# LoRA adapter release

## Artifact

The public adapter is packaged as `nanbeige4.2-3b-calibration-lora-v0.1.0-github-release.tar.gz` and should
be attached to the repository's `v0.1.0` GitHub release rather than committed to Git history.

| Item | SHA-256 |
| --- | --- |
| release archive | `94b3478e7920179b74af58e74297a47a16c515c542353cd5ae63b034d89cb578` |
| `adapter_model.safetensors` | `3675e7636b4a1c4a3667a40ca56c30d9fb7b7e2e7037c6bc603a27dc20eec661` |
| public `adapter_config.json` | `011c30c2b1224e3fed2ad4ba4fb72d22a3b58516f197cdb53b179355bc77d63f` |
| tokenizer configuration | `3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518` |

The safetensors hash is identical to the frozen H100 training output. The public copy changes no
weights; it only replaces the machine-local base-model path in the PEFT configuration with the
portable identifier `Nanbeige/Nanbeige4.2-3B` and replaces the generated placeholder README with a
complete research model card. The original server artifact remains preserved separately.
The release archive contains no AppleDouble files or macOS extended attributes.

## Load

Download the official Nanbeige base checkpoint separately, extract the release archive and load the
adapter through PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Nanbeige/Nanbeige4.2-3B"
tokenizer = AutoTokenizer.from_pretrained("release_adapter", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "release_adapter")
```

Use the adapter through the matching Quantum Calibration Agent native tool protocol. Loading the
weights alone does not provide controller bounds, deterministic analysis tools or IQ acceptance.

## Distribution boundary

Base-model weights are not duplicated in this repository or release. Preserve the official
Nanbeige4.2-3B license and attribution. No project-specific code/data license is inferred from the
base model's license; repository owners should add one explicitly if redistribution rights are to be
granted for project-authored source and data.
