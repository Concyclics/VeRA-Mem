"""Bounded single-GPU capacity probe; this is not a learning experiment.

Run only on an explicitly selected, available GPU. Synthetic token IDs and
writer features measure the existing BlockVeRA compute path, not its quality.
No datasets, weights, or dependencies are downloaded; no checkpoint is saved.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="One live GPU UUID")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not args.gpu.startswith("GPU-") or "," in args.gpu:
        parser.error("Specify one GPU UUID")
    if not 3 <= args.steps <= 20:
        parser.error("Use 3 to 20 measured steps")
    if args.output.exists():
        parser.error("Output already exists")
    if not (args.model / "config.json").is_file():
        parser.error("Local model config is missing")
    probe = subprocess.run(
        ["nvidia-smi", "-i", args.gpu,
         "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=15)
    gpu_name, used, total, utilization = [v.strip() for v in probe.stdout.strip().split(",")]
    if int(used) > 1024 or int(utilization) > 5:
        parser.error("Selected GPU is occupied; this probe does not wait or terminate processes")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    from torch.nn import functional as F
    import transformers
    from transformers import AutoModelForCausalLM
    from vera_mem.block_vera import BlockVeRA, _retrieve_block

    torch.set_num_threads(4)
    torch.manual_seed(20261013)
    torch.backends.cuda.matmul.allow_tf32 = True
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), *sorted((root / "src/vera_mem").glob("*.py"))]
    result = {
        "schema": "paper-scale-capacity-probe-v1", "complete": False,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Capacity and throughput only; no generalization or training-quality evidence",
        "gpu_name": gpu_name, "gpu_total_mib": int(total),
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "cpu_threads": 4},
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
        "settings": {"layer": 20, "rank": 64, "key_dim": 64, "slots_per_group": 3,
                     "training_bank_groups": 128, "block_mode": "block_outer",
                     "warmup_steps_per_case": 3, "measured_steps_per_case": args.steps,
                     "inputs": "Fixed synthetic token IDs and random detached writer features",
                     "injection": "Every position of the selected layer",
                     "supervision": "Next-token CE on latter half of each sequence plus 0.2 address CE",
                     "optimizer": "AdamW, lr=1e-4, adapter only",
                     "precision": "BF16 backbone, FP32 adapter, SDPA, no gradient checkpointing"},
        "excluded_costs": ["Corpus loading and tokenization", "Support encoding with the backbone",
                           "Global index building, search and refresh", "Distillation teacher forwards",
                           "Validation and generation", "Checkpoint IO and backup", "Multi-GPU communication"],
        "gpu_cases": [], "cpu_exact_search_kernel": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    persist()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    result["model_load_seconds"] = time.perf_counter() - start
    target = model.model.layers[20].mlp.down_proj

    for batch, length in [(1, 512), (1, 1024), (2, 1024), (1, 2048)]:
        torch.cuda.empty_cache()
        torch.manual_seed(20261013)
        reader = BlockVeRA(target.in_features, target.out_features, rank=64, key_dim=64,
                           temperature=0.05, seed=20261013, block_mode="block_outer").to("cuda")
        support_features = torch.randn(128 * 3, target.in_features, device="cuda")
        reader.fit_statistics(torch.randn_like(support_features), support_features)
        reader.fit_value_statistics(support_features)
        reader.set_feature_bank(support_features)
        del support_features
        optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-4)
        tokens = torch.randint(100, 12000, (batch, length), device="cuda")
        auxiliary = {}

        def hook(module, inputs, output):
            keys, values = reader.encoded_feature_bank()
            delta, info = reader(inputs[0], keys, values, return_info=True)
            similarities = info["queries"] @ keys.T
            logits = torch.logsumexp(similarities.reshape(batch, length, 128, 3) / 0.05, -1)
            auxiliary["address"] = F.cross_entropy(
                logits.reshape(-1, 128), torch.zeros(batch * length, dtype=torch.long, device="cuda"))
            return output + delta

        handle = target.register_forward_hook(hook)
        case = {"batch_size": batch, "sequence_length": length,
                "scored_tokens_per_step": batch * (length // 2),
                "input_positions_per_step": batch * length,
                "trainable_parameters": sum(p.numel() for p in reader.parameters() if p.requires_grad)}
        times, finite_losses = [], []
        try:
            for step in range(3 + args.steps):
                optimizer.zero_grad(set_to_none=True)
                if step == 3:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                output = model(input_ids=tokens, use_cache=False)
                logits = output.logits[:, length // 2 - 1:-1].float().reshape(-1, model.config.vocab_size)
                labels = tokens[:, length // 2:].reshape(-1)
                loss = F.cross_entropy(logits, labels) + 0.2 * auxiliary["address"]
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                finite_losses.append(bool(torch.isfinite(loss).item()))
                if step >= 3:
                    times.append(elapsed)
                del loss, logits, output
                auxiliary.clear()
            case.update(status="ok", step_seconds=times,
                        median_step_seconds=statistics.median(times),
                        scored_tokens_per_second=case["scored_tokens_per_step"] * len(times) / sum(times),
                        input_positions_per_second=case["input_positions_per_step"] * len(times) / sum(times),
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        all_losses_finite=all(finite_losses),
                        value_writer_gradient_nonzero=bool(reader.Wv.weight.grad.abs().sum().item() > 0),
                        query_gradient_nonzero=bool(reader.Wq.weight.grad.abs().sum().item() > 0),
                        backbone_has_gradients=any(p.grad is not None for p in model.parameters()))
        except torch.cuda.OutOfMemoryError:
            case.update(status="out_of_memory")
            # End the probe on OOM; no retry with changed scientific settings.
            result["gpu_cases"].append(case)
            persist()
            raise
        finally:
            handle.remove()
        result["gpu_cases"].append(case)
        print(json.dumps(case), flush=True)
        persist()
        del reader, optimizer, tokens, auxiliary, labels
        gc.collect()

    del model, target
    gc.collect()
    torch.cuda.empty_cache()
    # This is the exact-search tensor kernel, excluding store metadata and PCIe
    # transfers. It does not measure an ANN implementation or writing cost.
    for groups in (1024, 16384, 131072):
        keys = torch.randn(groups * 3, 64)
        values = torch.randn(groups * 3, 64)
        queries = torch.randn(1, 64)
        times = []
        with torch.no_grad():
            for step in range(8):
                began = time.perf_counter()
                output = _retrieve_block(queries, keys, values, 0.05)
                elapsed = time.perf_counter() - began
                if step >= 3:
                    times.append(elapsed)
        result["cpu_exact_search_kernel"].append({
            "groups": groups, "slots": groups * 3, "queries": 1,
            "kv_bytes": (keys.numel() + values.numel()) * 4,
            "latency_seconds": times, "median_seconds": statistics.median(times)})
        del keys, values, queries, output
        persist()
    result.update(complete=True, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    persist()


if __name__ == "__main__":
    main()
