# WikiText-103 first-round execution record

Status on 2026-10-13: **data and feature preparation completed; formal full-corpus training pending an available GPU**. This record contains preparation and engineering evidence, not a completed quality result. See the [registered protocol](wikitext_round1_protocol.md) for hypotheses, controls, and claim boundaries.

## Inputs and completed preparation

The full official WikiText-103 training split produced 124,245,019 text tokens, 124,273,491 causal LM targets, and 984,723 observed memory blocks. The final data manifest SHA-256 is `c1be8df1754b5fc4c81c3d69dceee2ac56e1a18daaf4932fab2600df55ef097c`.

Frozen feature extraction finished on H100. After correcting article boundaries, 977,745 exactly identical token sequences reused cached features and 6,978 blocks were re-encoded. Each reused row passed exact token-array equality after its hash lookup. The final feature manifest SHA-256 is `a5ecfd65f79d7fef4a1444d895f2ef789c34090c78f0ff5a0afdca873787a2a7`. Training-only statistics were recomputed over all corrected blocks.

The natural-passage episodes contain 16,384 training units, 512 validation units, and 1,000 test units from 58 non-overlapping test articles. Queries use two training styles and two held-out styles. This task is anchored three-word completion; it is not a general QA benchmark.

The Qwen3-4B-Instruct-2507 model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`. On the destination host, all three weight shards matched their pinned Hub SHA-256 metadata. All six final feature-cache files also passed full-file checksum verification after transfer.

## Engineering attempts

The initial CPU HNSW candidate index achieved only 80.31% recall@32 at `efSearch=512`, using actual layer-input queries with same-article exclusion. It delivered about 2,078 queries/s. The four-arm engineering run processed roughly 533–543 LM targets/s, with candidate search dominating the measured reader time. These measurements led to GPU IVF-Flat qualification before formal training.

That engineering run failed when its first contextual episode encountered a Transformers chat-template return-type change. The prompt builder now requests a token list explicitly and validates/normalizes the returned representation. No checkpoint from the failed engineering attempt is used to initialize formal training. Its logs remain in the private experiment archive.

The GPU IVF implementation passed a small integration check on an available local RTX 3090: build, exact-budget self-neighbor lookup, logical upsert, masking the old version, cloning, serialization, and restoration. This establishes API correctness for those operations, not million-block recall or training throughput. Full-corpus qualification and the 16-update training pilot remain mandatory.

Full Qwen prefix/suffix qualification on H100 measured zero hidden-state discrepancy against the ordinary projection hook and zero discrepancy for zero injection. All 20 corpus tests passed locally and in the H200 environment. They cover incremental KV caches, actual answer-token alignment, matched operator formulas, Q/K/V gradients, causal feature pooling, exact corpus target coverage, and article-clustered analysis.

## Authorized hardware change

Both H100 GPUs became occupied before the corrected GPU-index pilot. The user authorized switching to a verified free H200 on 2026-10-13. The destination uses the same model, data, feature cache, seed, losses, and planned full-pass budget. Frozen features remain the H100-produced artifacts; formal learner results will identify the actual training GPU.

The H200 environment matches the main H100 runtime versions: PyTorch 2.11.0 with CUDA 13.0, Transformers 5.5.4, NumPy 2.2.0, Hugging Face Hub 1.13.0, tokenizers 0.22.2, and safetensors 0.7.0. It uses the official Faiss GPU 1.15.1 package. Python differs (3.12 on H200, 3.10 on H100); the complete environment receipt records this distinction. The isolated environment does not modify other project environments.

At launch, all four H200 GPUs still had existing compute processes. A single durable execution controller is running and waiting for a genuinely free device from the four authorized H200 UUIDs. An idle utilization sample with allocated memory is insufficient. There is only one formal experiment; no duplicate runs are submitted to race for resources. It will qualify the index, complete the engineering pilot, run both training segments and evaluations, and generate the paired analysis.

The H100 preparation archive is fully backed up locally; all 11 cached feature files passed checksum verification. The H200 run has a separate local mirror that synchronizes every five minutes. A final remote artifact manifest and local checksum comparison are required after completion; an ordinary successful synchronization is not labeled final verification.

## Completion requirements

All three VeRA readers and parameter-matched LoRA must finish the same complete corpus pass. The 10M-token evaluation is diagnostic and does not select an arm or terminate training. Completion also requires full static evaluation, frozen-weight writes/edits/restoration/retention, article-paired analysis, and locally verified backups.

Until those artifacts exist, there is no new claim about WikiText PPL, improved generalization, successful continual learning at this scale, or superiority to Memory Decoder / MLP Memory. The preparation evidence above must not be presented as those results.
