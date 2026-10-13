# Scaling VeRA-Mem against Memory Decoder and MLP Memory

Review date: 2026-10-13. **A WikiText-103-scale experiment is a credible next target. The existing prototype is not yet a corpus trainer.** This review includes a completed hardware capacity probe, not a completed large-data learning experiment. The user confirmed that “MemoryEncoder” referred to **Memory Decoder**.

Start with approximately 100M corpus tokens and a measured one-pass budget, then consider 300M cumulative training tokens or the 600M-token Web setting. Do not start with the 300B-token general-memory experiment. Corpus size, cumulative supervised tokens, preprocessing work, and memory capacity must be reported separately.

## 1. Which published scale can we align with?

| Reference | Relevant experimental scale | Appropriate comparison |
| --- | --- | --- |
| [Memory Decoder, v2, Sections 3–4](https://arxiv.org/html/2508.09874v2) | WikiText-103, over 100M corpus tokens; GPT-2-XL builds retrieval supervision; a 124M GPT-2-small memory is trained. Domain experiments use a 0.5B memory. Reported setup: 8 A800 80GB GPUs. | Shared corpus, retrieval-distribution supervision, and language-model evaluation are practical reference points. |
| [MLP Memory, v3, Sections 4–5 and Appendix A](https://arxiv.org/html/2508.01832v3) | Scaling experiments use approximately 100M WikiText-103 and 600M mixed-Web tokens, with 124M–774M memory modules. QA uses 7B backbones and 1B memories trained on Wikipedia-2021. Reported setup: 32 A800 80GB GPUs. | The 100M setting is the first target; 600M is a later expansion. Its large QA model is a separate configuration. |
| [Memory Decoder at Scale, v1, Appendices A–B](https://arxiv.org/html/2607.27919v1) | General memories: 1.4B/2.8B/6.9B parameters, 300B training tokens over a 207B-token datastore, and 64/64/256 GPUs. Domain corpora: 125M legal, 552M biology, and 4.4B finance tokens; domain memory starts from Qwen3-1.7B-Base. | The domain-data range is informative. The general-memory pretraining run is outside the present plan. |

These are reported experiment setups, not minimum GPU requirements or measured GPU-hours. The two original papers describe compute budgets relative to one epoch of 7B-model training. That does not establish their wall time or mean that the memory itself was trained for one epoch.

The pinned [MLP Memory training example](https://github.com/LUMIA-Group/MLPMemory/blob/6801e43fc54a40f3217c2245a0c0019bbf137372/scripts/train_mlpmem_offline.sh) specifies **50 epochs** for a GPT-2-large memory; the [Memory Decoder example](https://github.com/LUMIA-Group/MemoryDecoder/blob/1aad7461e15ad4072edabf6b003d1578d31892dc/scripts/train_memdec.sh) specifies **70 epochs** for GPT-2-small. These are repository examples, not verified execution logs for every paper table. MLP Memory's 1B appendix instead lists five epochs and a maximum-sample setting; that cap is not an observed token count. A one-pass 100M-token run aligns a useful data scale, **not the papers' complete optimization budgets**.

MLP Memory's official implementation provides offline and online feature computation, IVFPQ indexing, and cached sparse retrieval targets. Precomputed Mistral features, token IDs, and memory weights cannot be reused unchanged for Qwen. See the [pinned implementation](https://github.com/LUMIA-Group/MLPMemory/tree/6801e43fc54a40f3217c2245a0c0019bbf137372).

## 2. What transfers to our research question?

Both original methods train a separate parametric predictor with next-token CE and forward KL to a retrieved next-token distribution, then interpolate its output probabilities with the base model. Memory Decoder reads tokens; MLP Memory reads backbone representations. Their published objectives support learning from many contextual retrieval examples rather than only fixed fact IDs. They do not establish our desired gradient-free online write behavior. [Memory Decoder method](https://arxiv.org/html/2508.09874v2#S3), [MLP Memory method](https://arxiv.org/html/2508.01832v3#S3)

Our proposed transfer is **corpus-trained shared Q/K/V writers and readers, followed by frozen-weight online VDB updates**. Corpus training should teach a reusable interface; new observations should become new memory records without changing the shared weights. This is a hypothesis requiring a separate continual-memory evaluation.

Current evidence does not already establish this at useful scale. The latest block study used 64 training entities, 128 three-word payloads, and 16-fact online banks. Across 18 models it accumulated 3,059,712 answer-supervised tokens, about 170K per model, with extensive repetition. Its preregistered continuation criterion failed despite partial improvements on novel combinations. The prior Wikipedia warm-up used only 8,192 training observations. A corpus containing millions of diverse contexts is a substantial change, not a longer run of the same task. See [block results](block_results.md) and [cold-start results](coldstart_results.md).

## 3. Completed capacity probe

The [probe script](../scripts/benchmark_scale_envelope.py) ran on one available H100 PCIe using the existing Qwen3-4B-Instruct-2507 copy, frozen backbone, layer-20 BlockVeRA, rank/key width 64, three slots, and 128 synthetic memory groups. Each case used three warm-up steps and eight timed optimizer steps. It injected at every sequence position and scored the latter half of each sequence. The workload used synthetic token IDs and writer features; **loss values are not capability results**.

| Batch × input length | Supervised tokens/s | Peak allocated GPU memory |
| --- | ---: | ---: |
| 1 × 512 | 5,574 | 9.32 GiB |
| 1 × 1024 | 7,946 | 10.97 GiB |
| 2 × 1024 | 8,267 | 14.15 GiB |
| 1 × 2048 | 8,094 | 14.15 GiB |

All four cases completed with finite losses, nonzero value-writer/query gradients, and no backbone gradients. The query gradient here includes address supervision; it does not show that hard selection receives an answer-loss gradient. Raw timings, environment versions, configuration and source hashes are in [capacity_probe.json](results/paper_scale/capacity_probe.json). Two preceding setup attempts stopped before optimization because initialization statistics were incomplete; their snapshots and logs were retained in the local backup. The completed probe fitted both required statistics. The selected GPU returned to zero allocated MiB after the probe exited.

To repeat the capacity check after installing the [tested environment](reproducibility.md), select an available GPU UUID and a new output file:

```bash
PYTHONPATH=src python scripts/benchmark_scale_envelope.py \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --gpu GPU-REPLACE-WITH-A-CONFIRMED-AVAILABLE-UUID \
  --output ../runs/new_capacity_probe/result.json
```

At the slower measured 1024-token rate, simple division gives these **training-kernel equivalents for one arm and one seed**:

| Cumulative supervised tokens | Kernel-equivalent hours |
| --- | ---: |
| 10M | 0.35 |
| 100M | 3.50 |
| 300M | 10.49 |
| 600M | 20.98 |

These are not end-to-end runtime predictions or guaranteed lower bounds. The short synthetic benchmark excludes support encoding, teacher forwards, global retrieval, data loading, checkpoints, evaluation, and backups; future readers may change its compute. A provisional scheduling envelope of **2–6 times the kernel figure** would allocate roughly **7–21 GPU-hours per 100M-token arm**, before separately measured one-time preprocessing. This multiplier is a planning assumption, not an observation. Three arms and three seeds multiply training work by nine. Do not assume multi-GPU linear speedup.

The live H100 session exposed two GPUs, with one occupied and one available at inspection, rather than the four GPUs in the older inventory. The selected data filesystem had approximately 599 GiB free. These are availability observations, not a reservation. Private process/UUID details remain outside the public repository.

### Retrieval and storage are the immediate engineering constraints

The current CPU exact-search kernel took median 0.30 ms, 3.26 ms, and 27.73 ms per query for 1,024, 16,384, and 131,072 three-slot groups, respectively, using four CPU threads. This excludes metadata lookup and GPU transfers. It is enough to reject treating the current per-token full scan as a scalable implementation; an ANN backend has **not** been benchmarked yet.

For a nominal 100M-token corpus, the following sizes are arithmetic, not measured artifacts:

| Representation | Payload size | Consequence |
| --- | ---: | --- |
| Every layer-20 down-projection input, 9,728 dimensions, FP16 | 1.946 TB | Does not fit the inspected free space. |
| Every 2,560-dimensional hidden state, FP16 | 512 GB | Leaves inadequate room for a comfortable full preprocessing pipeline. |
| Eight pooled 9,728-dimensional features per 128-token chunk, FP16 | 121.6 GB | A candidate writer-training cache; needs measured quality and IO. |
| Eight K/V rows per chunk, each with 64+64 dimensions, FP16 | 1.6 GB | Only projected payloads; excludes indices, metadata and training features. |
| Sparse teacher targets averaging 64 token/probability pairs, plus offsets/labels | 39.6 GB | Assumes 4-byte IDs, 2-byte probabilities, 8-byte offsets and 4-byte labels; actual sparsity is unknown. |

The eight-slot representation is a **proposed extension**; today's reader is hard-coded to three slots. FP16 storage also needs comparison with FP32 retrieval. Do not create dense full-vocabulary teacher caches. Use sharded features and sparse targets, measure temporary-file peaks, and validate a projected disk budget before preprocessing. [estimate.json](results/paper_scale/estimate.json) contains the arithmetic.

## 4. Proposed first experiment

This is a design, not a sealed preregistration or an implemented corpus launcher. The machine-readable [plan](results/paper_scale/plan.json) records that boundary.

**Data and model.** Keep Qwen3-4B-Instruct-2507 frozen for continuity. Use the official [WikiText-103 raw splits](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Reconstruct article boundaries before chunking, preserve official train/validation/test separation, and record Qwen-tokenized counts. The conventional “100M” size is not necessarily 100M Qwen subword tokens. Start with one complete training pass, using a 10M-supervised-token development checkpoint; report actual exposure rather than silently repeating data to hit a nominal count.

**Writer and retrieval.** Implement a separately versioned corpus interface: initially consider 128-token supports, eight contiguous pooled slots, rank/key width 64, an ANN shortlist of 32 blocks, and exact reranking to four complete blocks. These are proposed settings to qualify in the pilot. Preserve source/chunk IDs and writer-version IDs. Rebuild projected keys whenever offline writer changes invalidate the index; measure refresh cost. Online inference uses a fixed writer and supports append/upsert. The actual VeRA-layer input remains the query source; no question parser or external semantic embedding replaces it.

**Matched readers.** Compare diagonal, a factor-mean rank-one control, and the full block operator using identical data, candidates, Q/K supervision and memory bytes. If selected rows have factors `u_s,w_s` and mean value `m`, compare `diag(m)`, `diag(m)+mean(w_s)mean(u_s)^T`, and `diag(m)+mean(w_s u_s^T)`. The last two use the same factor maps. This improves on the historical pooled-value control, where averaging before nonlinear normalization confounded the rank comparison. Use the same group/row weighting in all arms. Any content-dependent weighting or multi-block selection is a new mechanism and must be identified as such; this is not a pure “more data” ablation.

**Training.** Use two explicitly accounted components: corpus next-token CE plus sparse retrieval-distribution KL, and write/read episodes with observed support text and independently expressed queries. The KL is applied to the vocabulary distribution produced after our adapter; it does not add a standalone 1B vocabulary predictor. Episodic contextual teachers can supply answer-distribution supervision. Teacher-forced targets are offline distillation; on-policy distillation would additionally require student-generated rollouts and their measured cost. Record LM and episode tokens separately. Retain a discriminative address loss because hard candidate selection alone does not train unselected Q/K addresses. Include empty-memory examples and hard negatives matched in style but differing in content. Qualify natural support reconstruction before committing to a full pass. Loss weights, mixture, teacher quality, index type, and refresh interval must be finalized on development data and registered before confirmation.

**Baselines.** Include the frozen base, retrieved-text context, and LoRA with approximately matched trainable parameters and measured update cost. For an actual head-to-head claim against MLP Memory or Memory Decoder, train compatible adaptations on the same Qwen tokenization/corpus or run a separate faithful GPT-2 reference track. The former changes the published setup; the latter does not isolate Qwen architecture effects. Existing Mistral/GPT-2 checkpoints and their perplexities are not same-backbone baselines. Report trainable parameters, stored bytes, preprocessing cost and inference latency separately; matching 100M data tokens does not make a 2M interface parameter-matched to a 1B memory.

### Keep two evaluations separate

| Evaluation | Memory access rule | What it can establish |
| --- | --- | --- |
| Static corpus language modeling | Train-only memory; do not ingest validation/test documents. Use 1024-token windows with 512 scored tokens and deterministic boundary/masking rules. | Held-out NLL/PPL change on a fixed memory, with matched tokenizer and base. The window follows the original Memory Decoder evaluation. |
| Continual write/read | Freeze shared weights and statistics. Observe a new support, write it, then ask held-out queries; no future supports or query labels enter the store. | Novel-entity/content transfer, paraphrase robustness, update/restoration, and retention after subsequent writes. |

For corpus retrieval targets, exclude the exact query record and overlapping duplicate spans; use a stricter same-article exclusion as a declared variant. Never permit future text from the query document to leak into ordinary next-token evaluation. In explicit memory episodes, support text may contain the answer because it is the observation being remembered; gold QA labels must not generate the writer features. Keep that protocol separate from static PPL.

The continual evaluation should use at least 1,000 fresh held-out support/query units, multiple query styles, and a stream of successive writes. Include empty memory, shuffled values, stale values, correct-block oracle, and restored-record controls. Report free-generation EM/F1, retrieval recall, joint update-and-restore accuracy, old-fact retention, and irrelevant-memory impact. Bootstrap by source entity/article, not individual paraphrase. Facts absent from adapter training are not necessarily absent from Qwen pretraining: use pre-write baselines and controlled counterfactual updates to establish the effect of the memory.

## 5. Progression and decision rules

1. **Complete:** capacity probe and literature/code audit. Probe artifacts were copied back and checksum-verified locally. No corpus training, teacher cache or large ANN index was created.
2. **Implement and qualify:** document packing, span pooling, generalized block reader, sparse targets, indexed retrieval, index versioning, paired evaluation and complete cost logging. Existing `block_run` cannot perform this proposed run by changing its step count.
3. **First large-data run:** one full WikiText-103 pass for the primary block recipe, with a fixed 10M-token development checkpoint. Proposed continuation rules: real memory beats empty and shuffled memory on held-out new-support EM by at least 10 percentage points with a positive lower bound of a 95% entity-level bootstrap confidence interval; real retrieval recovers at least half of the positive oracle EM improvement over empty memory; empty/irrelevant-memory validation PPL worsens by no more than 2% relative. These are our proposed gates, not thresholds established by either paper. Register final gates and fresh confirmation identities before running.
4. **Comparison and replication:** complete the two matched reader controls at the same exposure, then repeat all three arms with three seeds. Extend to 300M cumulative tokens or a 600M independent corpus only after a reproducible useful signal. More epochs and more distinct documents answer different questions.

If the oracle fails, investigate compression/readout before increasing the index size. If the oracle works but retrieval fails, focus on Q/K and the index. If static PPL improves but new-support writing does not, report successful corpus adaptation without claiming continual learning. The present evidence supports attempting the 100M-scale study; it does not predict that scaling will resolve generalization.
