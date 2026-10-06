# VeRA-Mem: literature, research questions, and experimental design

Review date: 2026-10-05. Base model: `Qwen/Qwen3-4B-Instruct-2507`.

This document defines research designs and interpretation boundaries; it does not report experiments that have not happened. Each run's configuration, individual outputs, and results establish the implementation, executed scale, and outcome. Branches labeled “future” are not implemented or validated merely because they appear here.

The main architecture is a **memory-conditioned VeRA extension**: the current input to a VeRA layer produces a query for sparse value retrieval from a vector database (VDB). Retrieved values serve as rank-dimensional VeRA parameters in the current token's computation. A complete new observation generates keys/values from the same layer's input features, and committing them to the VDB enables later inference. The underlying parameterization comes from [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454); dynamic retrieval and online database writes are project extensions to be tested.

## 1. Research questions

The project asks: **With the language-model backbone and VeRA random matrices frozen, can vectors retrieved per token within a layer dynamically parameterize VeRA, while input-generated keys/values continually update external memory, outperforming simple controls under explicit storage, read/write-cost, and capability-retention constraints?**

Separate four easily conflated questions:

1. **Writing:** after receiving a correct new fact, can the model answer immediately?
2. **Retention:** after additional facts are written, can it still answer earlier ones?
3. **Addressing:** can it retrieve evidence from the correct storage unit, rather than succeeding only when the correct location is supplied?
4. **Readout:** after retrieving the correct latent vector, can it decode the answer and handle new question wording?

A decrease in one training loss directly supports only part of the first question. Same-question retest gains are not question-format generalization; gains from more adapters do not by themselves establish useful routing. Candidate contributions are writable dynamic VeRA parameter vectors, their interaction with token inputs, and reproducible cost–retention trade-offs. The combination of low-rank adaptation, retrieval, and CPU storage must not itself be claimed as unprecedented. LoRA banks and additive latent readers are controls or future extensions, not substitutes for the main architecture.

## 2. Closest literature and design implications

Only directly relevant mechanisms are summarized below. A paper's hardware speed, training scale, or model score is not a performance forecast for this project.

| Work | Existing mechanism | Implication or constraint |
| --- | --- | --- |
| **VeRA**, 2023/2024 | Shared frozen random A/B matrices, with trainable b/d vectors parameterizing a low-rank branch | This project replaces the fixed rank vector d with aggregated VDB values and learns input-to-query/key/value mappings. This extends original VeRA; online memory is not an existing feature of the original paper. [Paper](https://arxiv.org/abs/2310.11454) |
| **GRACE**, 2022/2023 | A discrete hidden-space key–value editing codebook uses local retrieval to modify outputs while freezing backbone weights | Hidden-state queries to external writable memory have direct precedent. Evaluate locality, paraphrases, and sequential edits, not just memorized training questions. [Paper](https://arxiv.org/abs/2211.11031), [author code](https://github.com/thartvigsen/grace) |
| **Larimar**, 2024 | Adds a trained episodic-memory controller supporting rapid reads, writes, and fact modification | One-shot writing does not mean no offline training. Its trained read/write interface motivates our proposed frozen-backbone/interface separation, with interface-training costs counted separately; this is not a claim that Larimar's decoder stays frozen during its own training. [Paper](https://arxiv.org/abs/2403.11901), [author code](https://github.com/IBM/larimar) |
| **WISE**, 2024 | Separates original parametric memory from editing-side memory; routing, knowledge shards, and merging support continual editing | Multiple-slot isolation, retention of the original model branch, and routing gates must be positioned relative to existing editing systems. Parameter sharding is not automatically external vector memory. [Paper](https://arxiv.org/abs/2405.14768) |
| **M+**, 2025 | Builds on MemoryLLM with CPU long-term latent memory and a jointly trained retriever; separate LoRA modules handle reads/writes, and retrieved memories enter generation through attention | A close precedent combining external latent memory, LoRA read/write interfaces, and CPU storage. Here values modulate VeRA parameter vectors; the distinction is the readout interface and budget, not the first long-term vector bank. [Paper](https://arxiv.org/abs/2502.00592), [author code](https://github.com/wangyu-ustc/MemoryLLM) |
| **Parametric RAG / PRAG**, 2025 | Parameterizes document knowledge in FFN adapter parameters and retrieves that parametric knowledge at query time | Storing document/fact adapters and activating or combining them after retrieval is an existing direction. Compare write cost and multi-adapter interference. [Paper](https://arxiv.org/abs/2501.15915), [author code](https://github.com/oneal2000/PRAG) |
| **Doc-to-LoRA**, 2026 | Meta-trains a hypernetwork that maps new context to LoRA in a forward pass, approximating context distillation | Closely related to a future vector-to-low-rank-parameters branch. Distinguish per-fact gradient writes from forward-pass writes after offline training; online timing must not hide meta-training costs. [Paper](https://arxiv.org/abs/2602.15902), [author code](https://github.com/SakanaAI/Doc-to-LoRA) |
| **Understanding LoRA as Knowledge Memory: An Empirical Analysis**, 2026 | Studies LoRA memory capacity, knowledge internalization, and multi-module composition; the reviewed arXiv version is v5 dated 2026-07-29 | Directly relevant to capacity, rank, and multi-module design. Match total capacity and separate oracle from actual routing rather than reporting only a multi-LoRA total score. [Paper](https://arxiv.org/abs/2603.01097) |

### 2.1 Three meanings of memory

| Category | What changes | Relationship to this project |
| --- | --- | --- |
| Appendable external memory | Database text, vectors, or cells | Appending a record is online nonparametric storage; participating in inference does not automatically make it test-time training. |
| Supervised writing / online adaptation | After a correct answer is revealed, write an input-encoded vector; controls may instead update parameters by gradients | The former needs no online SGD but still uses label supervision. It suits controlled associations/labeled edits; specify exactly when the answer becomes available. |
| Within-sequence self-supervised fast weights | A hidden-state model is updated using a self-supervised objective defined on the observable sequence | Closer to TTT/Titans, requiring its own objective, causal protocol, and outer-loop training. |

**TTT** represents recurrent state as a model and updates that state through self-supervised learning steps. **Titans** uses learnable neural memory with association-loss-related updates and forgetting. Our main architecture freezes the shared network online and appends input-encoded vectors, making it writable external memory. If observations include newly revealed gold answers, call it supervised writing—not unsupervised TTT merely because no online SGD occurs. [TTT](https://arxiv.org/abs/2407.04620), [Titans](https://arxiv.org/abs/2501.00663)

**Engram** uses deterministic token-n-gram addressing and a statically trained memory table; host-memory prefetch benefits from addresses being known early. Semantic nearest-neighbor retrieval whose query becomes available only after a hidden state is computed has a different dependency chain. Engram's low-overhead conclusions cannot be transferred directly. [Engram](https://arxiv.org/abs/2601.07372)

### 2.2 Additional reading order

1. Low-rank parameterization and parameter efficiency: [LoRA](https://arxiv.org/abs/2106.09685), [VeRA](https://arxiv.org/abs/2310.11454). VeRA's frozen random matrices still occupy runtime memory; trainable-vector size alone is not total system cost.
2. External retrieval memory: [kNN-LM](https://arxiv.org/abs/1911.00172), [Memorizing Transformers](https://arxiv.org/abs/2203.08913). Output-distribution interpolation, attention-KV retrieval, and FFN-residual injection are different readout mechanisms.
3. Representation stability: [LongMem](https://arxiv.org/abs/2306.07174), [MemoryLLM](https://arxiv.org/abs/2402.04624). Whether old vectors remain retrievable after encoder changes is separate from capacity.
4. Trained sparse memory tables: [Memory Layers at Scale](https://arxiv.org/abs/2412.09764). A large pretrained table is not equivalent to episodic memory appended at runtime.
5. Parameter generation: [Text-to-LoRA](https://arxiv.org/abs/2506.06105). Task-description-to-adapter and fact-content-to-memory-parameter training target different objects and should be compared separately.

## 3. Shared model, data, and execution conventions

Fix the base to `Qwen/Qwen3-4B-Instruct-2507` and record the actual Hugging Face commit revision, weight/tokenizer sources, software versions, and precision. Use official `tokenizer.apply_chat_template` rather than handwritten control tokens. This is a non-thinking model; evaluation must not rely on extracting reasoning text. [Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

The official configuration has 36 Transformer layers, hidden size 2560, and MLP intermediate size 9728, so one `mlp.down_proj` maps `9728 → 2560`. Standard single-layer LoRA has `r × (9728 + 2560) = 12288r` parameters, excluding extra bias, routing, and read/write networks. Layer indices use 0-based Python indexing. [Configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)

The experiment manifest should lock:

- `data_seed`, `initialization_seed`, write order, and entity/session splits; formal runs use at least three random seeds.
- Insertion layer, rank, query/key dimension, top-k, temperature, support-feature position/pooling, write/update rules, offline loss weights, learning rates, maximum sequence length, and generation settings. Controls additionally record slot count and online update steps.
- Raw data revision/SHA256, conversion-code commit, model revision, configuration hash, and actual processed sample IDs.
- Independent parameter, optimizer, and memory initialization for each method; sequential runs must not implicitly inherit a previous method's state.

Training loss covers only assistant answers and the specified termination token; prompt/padding labels are `-100`. Check actual supervised tokens, shifted labels, and template boundaries on small examples. Use fixed greedy decoding and length limits, retaining both raw text and normalized answers. Count format errors separately rather than concealing them with permissive substring matches.

## 4. Main architecture: input-driven VDB–VeRA reads and writes

### 4.1 Parameterization and per-token, within-layer reading

Let `x_t ∈ R^din` be the selected linear layer's input and `W ∈ R^(dout×din)` its frozen weight. For Qwen `mlp.down_proj`, `din=9728` and `dout=2560`. Here x is the actual down-projection input, not a substituted 2560-dimensional residual hidden state. Define:

```text
A ∈ R^(r×din), B ∈ R^(dout×r)          Frozen, randomly initialized
b ∈ R^dout                             Shared output-scaling vector
Wq, Wk ∈ R^(dk×din), Wv ∈ R^(r×din)   Shared read/write mappings
M^(s) = {(k_i, v_i, metadata_i)}         Read-only VDB snapshot, version s
```

Original VeRA uses `delta = b ⊙ B[(A x) ⊙ d]`, where d is a shared adapter parameter. This project replaces d with retrieved rank vector `vbar_t`:

```text
q_t = l2_normalize(Wq · norm(x_t))
I_t = topk_i cosine(q_t, k_i),  i ∈ M^(s)
alpha_ti = softmax_i(cosine(q_t, k_i) / temperature), i ∈ I_t
vbar_t = Σ_{i∈I_t} alpha_ti · v_i

delta_t = b ⊙ B[(A x_t) ⊙ vbar_t]
y_t = W x_t + delta_t
```

`norm` is configured feature normalization with a numerical-stability term, distinct from cosine L2 normalization. Save A/B random seeds, initialization distributions, and precision. Empty VDB means `vbar_t=0`, returning exactly to the frozen backbone without hiding another nonzero static adapter. Any extra amplitude coefficient must appear explicitly in formulas, configuration, and logs.

**Queries must be generated inside the VeRA-layer hook from each token's actual input.** Prefill may batch tokens; each new decode token refreshes q, top-k, and value aggregation. Retrieving once from a question-level vector and broadcasting one value over the answer is not the main architecture. It is a separate question-cache ablation.

A response fixes `M^(s)`, not `I_t`: retrieval can change by token, but memory contents and shared weights remain constant while reading. Previously generated KV under that snapshot can be used autoregressively. After committing new memory, new requests should create new context; do not unconditionally reuse request KV across memory versions.

### 4.2 Input-driven writing and online continual memory

After a complete observation arrives, obtain `support_x ∈ R^din` at the same layer using the frozen backbone. The first version uses a prespecified terminal position or pooling rule and saves its definition. Disable adaptation during support extraction for stable original-input features. Pooling alternatives are ablations and must not be chosen after inspecting test questions.

```text
After the observation is complete:
  k_new = l2_normalize(Wk · norm(support_x))
  v_new = tanh(Wv · norm(support_x))        # r-dimensional VeRA parameter vector
  pending = (k_new, v_new, event_id, timestamp, encoder_version)

Atomic commit:
  append_or_update(M^(s), pending) → M^(s+1)
```

Online, freeze the backbone, A/B, b, and Wq/Wk/Wv. Continual learning consists of appending/updating input-generated memory entries, not gradient-optimizing shared weights for each test fact. Fix append/update semantics: new events can append; corrections to the same event can update using a legitimate event ID; historical-version retention depends on the temporal-query protocol. Evaluation answers must not choose the record to update.

Commit only after the observation completes; a new record must not affect its own pre-observation prediction. In synthetic/MedMCQA supervised tracks, observations may contain subsequently revealed answers. In real-conversation tracks, they contain only text/dates already received. Storing an answer as part of a past observation is allowed; placing a future gold answer in the current query is leakage.

Initially use one VeRA insertion layer so its upstream backbone remains frozen. With future multilayer memory, later inputs may already depend on earlier memory injections. Representation changes across snapshots then require separate handling; frozen parameters alone do not guarantee stable multilayer features.

### 4.3 Offline episodic training: learn addressing and readout together

Construct support/query episodes on training entities, with final-test entities disjoint from train/dev. Supports contain only legitimate episode observations; queries use another formulation of the fact. Train b and Wq/Wk/Wv while freezing the backbone and random A/B:

```text
L = L_answer_LM + lambda_align · L_query_support_contrastive
                    + optional, prespecified regularization

L_query_support_contrastive:
  Bring the target-layer query from the question prefix closer to the matching support key;
  use other entities/attributes in the same episode as negatives.

L_answer_LM:
  Predict answers using actual within-layer retrieval and VeRA parameter modulation;
  apply causal LM loss only to the assistant answer and specified termination token.
```

Contrastive alignment uses the final valid query before the question/answer boundary or a prespecified question position, never a token after the complete answer. Teacher-forced LM training uses a standard causal prefix for next-token prediction; future answer tokens are invisible. Test generation uses only the current question and generated prefix. Main retrieval diagnostics are measured before answer generation so teacher-forced answer-prefix assistance is not mixed into answer-free retrieval scores.

Hard top-k record IDs are nondifferentiable, but selected values, scores, and downstream modulation must retain gradient paths. **With top-k=1, softmax is always 1; LM loss has essentially no addressing gradient through that weight, making explicit contrastive alignment particularly important.** Do not detach trainable writer outputs offline and expect Wv to learn. Online commits, by contrast, should detach and save stable vectors.

Freezing backbone weights does not mean wrapping the whole post-injection forward pass in `no_grad`, which would prevent LM-loss learning of shared parameters. b may start at zero to preserve base behavior, but the value branch must not also be exactly zero so that every LM gradient is blocked. Record parameter-group gradient norms. Report joint training separately from alignment-only and LM-only controls.

### 4.4 Current implementation: exact CPU VDB and sparse result transfer

Pilot keys/values and metadata are actually stored on CPU in `PersistentVectorDB`; online reads have no full GPU index mirror. At each token, the VeRA hook sends its query to CPU for exact cosine top-k and softmax-weighted value mixing, then returns the rank vector to GPU for VeRA. Prefill can batch queries; decode repeats lookup for each new token. Reads of a snapshot do not interleave with uncommitted writes.

This is a small, synchronous exact-CPU reference implementation. **It neither validates large-scale CPU offload nor establishes negligible CPU retrieval/transfer cost.** Exact scans grow with record count and key dimension; per-token synchronization overhead must be measured. Temporary GPU support tensors for differentiable offline episodes are not an online VDB mirror. ANN, GPU index mirrors/caches, and prefetch are separate future comparisons, with retrieval, transfer, and modulation costs decomposed.

### 4.5 Required mechanism controls and architectural boundaries

Compare **oracle records, actual per-token retrieval, empty memory, random wrong records, and shuffled values** using the same written memory. Oracle may select only legitimately written supports and is an ideal-evidence diagnostic, not a guaranteed numerical performance upper bound. Do not insert answer tokens directly as values or invent new evidence. Wrong-record controls match record count/dimensions. The main method remains actual query-based retrieval; oracle scores cannot replace its final score.

If oracle fails, prioritize support encoding, parameter-vector capacity, and VeRA readout. If oracle works but real retrieval fails, prioritize Wq/Wk alignment. Unchanged performance after shuffling values does not establish memory use. If empty memory differs from the frozen backbone, check implementation or hidden adapters first.

| Control/extension | Difference from main architecture | Purpose |
| --- | --- | --- |
| Original static VeRA | d is a learned fixed vector; no VDB | Separate ordinary adaptation from dynamic external memory |
| VeRA parameter bank | Stores directly optimized b/d vectors; query selects a slot | Test parameter isolation; distinct from generating new values through Wv |
| LoRA bank | Trainable A/B, supervised gradient updates by slot | Capacity/interference control; not a replacement implementation of VeRA-Mem |
| Additive latent reader | `delta=B m`, without `(A x)⊙vbar` interaction | Test whether multiplicative parameterization is needed; separate baseline |
| Question-cached retrieval | One retrieved value reused throughout | Quantify per-token addressing cost/benefit; explicitly differs from the main setting |
| Dynamic output vector/hypernetwork | Generates b or additional weights too | Future extension; separately account for storage/training and compare with Doc-to-LoRA |
| Association-loss fast-weight state | Updates bounded state online rather than only appending to VDB | Future independent branch; do not label current external memory as TTT |

## 5. Three-stage data plan

### Stage I: controlled new associations and causal unit experiments

Generate random entity–attribute–value mappings through independent permutations; entity IDs must not encode answers. Split offline b/Wq/Wk/Wv train, dev, and final evaluation by disjoint entities/combinations. Each episode includes factual support, another question formulation, and negatives from other entities. Final-test facts enter only through online input-derived key/value writes and never shared-parameter updates. Separate writing templates from evaluation paraphrases.

Start with 32–64 new facts, 4 write blocks, and independent unwritten entities; expand formal scales to 128, 512, and 2048 facts. Small runs expose defects, not long-term capability. Cover at least:

- Original-question retests and independent wording; for multiple choice, permute options and update labels consistently.
- New values for the same entity/attribute, separately asking for the latest value and past time points, so retaining correct history is not mislabeled as update failure.
- Similar entities, shared attributes, and irrelevant writes to distinguish semantic confusion from limited capacity.
- Unwritten entities and missing attributes to check abstention and erroneous cross-association.

Use natural words and random strings as separate difficulty levels. Exact multitoken random-string generation is harder; report lengths/tokenizer distributions rather than treating arbitrary-code generation failure as failure of all memory mechanisms. For long-context stress, RULER offers multi-key, variable-tracking, and compositional-retrieval settings. [RULER](https://arxiv.org/abs/2404.06654)

### Stage II: supervised continual writing on MedMCQA

MedMCQA supplies domain multiple-choice questions and explanations for labeled knowledge writing and capability-interference tests. Pretraining exposure cannot be excluded, so it complements synthetic novel associations. [Paper](https://arxiv.org/abs/2203.14371), [data card](https://huggingface.co/datasets/openlifescienceai/medmcqa), [author code](https://github.com/medmcqa/medmcqa)

Construct a write stream from a fixed train subset and tune on independent development examples. Preserve an untuned validation control set. Initial runs can use 64–128 written questions and 32–64 controls, scaling later. Deduplicate by normalized question+options hash and retain source IDs. Verify the actual `cop`-to-A–D schema mapping at load time rather than trusting data-card prose. Filtering to single-choice questions can simplify the first run.

Main read inputs contain only question and options. If explanation `exp` is written, provide the same explanation to every method and label a separate QA+explanation-writing condition. A second pass over the same questions measures retention. Only unseen paraphrases, option permutations, or independent capability controls support their respective generalization/interference claims.

### Stage III: real multisession memory

Prioritize LongMemEval cleaned v1, pinning revision and file SHA256; the official repository points to cleaned data at the time of review. Validate the pipeline on a dev subset, then freeze the protocol for untuned formal evaluation. `oracle` and S full-history retrieval are separate tracks with separate scores; do not mix versions after updates. [Official repository](https://github.com/xiaowu0162/LongMemEval), [cleaned data](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), [paper](https://arxiv.org/abs/2410.10813)

Write historical text in true temporal order. Evaluation `answer`, `answer_session_ids`, and evidence labels are only for scorers/retrieval diagnostics; do not reverse-engineer writing material from evaluation QA. Reset memory per question. When multiple questions share one user's history, explicitly group by user and use consistent snapshots. First compare text RAG, oracle text, and latent memory, then inspect extraction, cross-session, temporal, updating, and abstention categories.

Use LoCoMo for later external validation, splitting/aggregating by complete conversation so correlated questions from one history do not cross dev/test. State explicitly if only its text portion is used. [Author repository](https://github.com/snap-research/locomo), [paper](https://arxiv.org/abs/2402.17753)

The public repository retains download/conversion code, versions, sample manifests, and aggregate results. Third-party redistribution follows licenses; private conversations do not enter public test sets.

## 6. Minimal fair controls and cost accounting

| Control | Single-layer down_proj configuration | Shared/total trainable parameters | Variable information activated per read | Question |
| --- | --- | ---: | ---: | --- |
| Frozen | No updates or memory | 0 | 0 | Original capability/format baseline |
| Text RAG | Fixed untrained retriever, top-k, token budget | 0 | Retrieved text | Whether simple external memory suffices |
| Text oracle | Correct evidence from legitimate history | 0 | Oracle text | Whether sufficient evidence makes the question answerable |
| Static VeRA | One b/d, fixed random A/B | `dout+r` | Fixed b/d | Parameterization baseline |
| VeRA-Mem | Shared b/Wq/Wk/Wv + input-derived VDB | `dout+din×(2dk+r)`, without biases | Per-token top-k, weights, rank vector | Main architecture |
| VeRA-Mem oracle | Same checkpoint and VDB | Same as main method | Correct support's value | Ideal-evidence readout diagnostic |
| LoRA-1 r4 | 1 slot × r4 | 49,152 | 49,152 active LoRA parameters | Low-budget single module |
| LoRA-4 r1 | 4 slots × r1 | 49,152 | 12,288 active LoRA parameters | Sharding at equal total parameters |
| LoRA-1 r16 | 1 slot × r16 | 196,608 | 196,608 active LoRA parameters | Higher-capacity single module |
| LoRA-4 r4 | 4 slots × r4 | 196,608 | 49,152 active LoRA parameters | Same activation size as one r4 module, with larger total capacity |

Adding Wq/Wk/Wv means the main architecture cannot retain original VeRA's `dout+r` parameter count. With `dk=r=64`, `din=9728`, and `dout=2560`, it learns **1,870,336** shared parameters, plus **786,432** frozen A/B elements and record-dependent keys/values. Count added biases, trainable norms, or projections too. Freezing shared weights online does not erase their offline training cost.

Active parameters are only a compute proxy, not actual FLOPs or wall latency. Per-token Wq, retrieval, write-time Wk/Wv, indexing, and text processing cost extra. Equal total parameters and equal active computation are different fairness axes, usually not satisfied by one control. Because VeRA-Mem queries vary by token, it cannot claim static VeRA's one-time merge into W with zero extra inference overhead.

First compare real/oracle/empty/shuffle VeRA-Mem reads at the same r, dk, and shared checkpoint, avoiding capacity confounds. LoRA-bank controls may compare hash, frozen-feature, and oracle routing. Give methods equal numbers of hyperparameter trials on the same dev set; forcing identical learning rates does not ensure fairness. Record offline tokens/steps, online information, update rules, total write seconds, and quality–time curves.

Separate storage into shared trainable parameters, frozen random matrices, per-fact keys/values, source text/metadata, authoritative CPU VDB, GPU query/result buffers, indexes, optimizer state, checkpoints, and peak CPU/GPU memory. Values have dimension r. With `bytes_per_element` bytes per element, raw-vector storage is `N × (dk+r) × bytes_per_element`. Measure index/metadata/transfer overhead separately; this formula is not total memory. Faiss `IndexFlatIP` stores float32 vectors, so use actual index dtype even if source files are half precision. [Faiss index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

## 7. Causal protocol: read, write, then read-only evaluation

```text
Offline: train b/Wq/Wk/Wv on episodes from independent training entities
         select configuration with independent dev entities; freeze and save shared parameters
Online initialization: load frozen backbone/shared checkpoint; start with empty test memory

For each time block t:
  1. pre_update: fix the memory snapshot from before t; retrieve per token within the layer and predict
  2. write: after full observations/labels are revealed, extract support_x and generate keys/values
  3. commit: only now append/update the authoritative VDB; record its version
  4. immediate: evaluate this and earlier blocks read-only on the new snapshot, forming retention matrix R
  5. controls: evaluate unwritten, unrelated, and newly worded examples; no optimizer updates

End: save parameters/memory; reload; run delayed evaluation entirely read-only
     verify unchanged parameter/memory checksums before and after evaluation
```

Queries come from the hook's current causal token inputs: before the answer boundary, only the question is present; during generation, only generated tokens are appended. Support features extracted after a complete observation may encode legitimately revealed answers, but enter only the write path. Neither support_x nor the final token of a complete question–answer pair can be used as a pre-answer query. One question vector must not replace per-token within-layer queries.

Keep offline shared-network training separate from online testing. b/Wq/Wk/Wv can learn how to write, address, and read on training entities; test content enters only through specified observation-based key/value commits. Updating shared parameters with test answers invalidates an unseen-entity generalization claim. Read-only evaluation checks both shared weights and CPU-VDB versions; absence of optimizer calls alone does not exclude implicit writes.

## 8. Metrics, statistics, and diagnostics

### 8.1 Quality metrics

- **Synthetic tasks:** exact match of complete normalized values, separately for original and paraphrased questions. Multiple-choice versions also report candidate log-probability accuracy and free-generation invalid rate. Substring hits are auxiliary diagnostics only.
- **MedMCQA:** strict A–D accuracy is primary, with candidate-ranking accuracy, option-permutation tests, and independent capability-control changes. Generated-answer and candidate-ranking accuracy are different metrics.
- **Conversation tasks:** report official task categories, distinguishing local EM/F1, human verification, and official judges. Do not call a score official without running the official judge. External paid judges require separately recorded model versions, costs, and authorization.
- **Retrieval:** evidence recall@k and MRR at the pre-answer boundary token, supplemented by per-token generation hit/switch traces and oracle–real gaps. Multi-evidence tasks report both any-hit and all-required-evidence recall. Teacher-forced answer-prefix retrieval must not be presented as answer-free retrieval performance.

Accumulate negative log likelihood over supervised answer tokens:

```text
token_weighted_NLL = Σ_examples Σ_answer_tokens (-log p(token)) / Σ_examples n_answer_tokens
corpus_PPL = exp(token_weighted_NLL)
```

Arithmetic mean of per-question PPL is not corpus PPL; long tails change its interpretation. Fix and disclose whether the main loss includes the termination token. RAG and closed-book answer metrics can be compared, but full-sequence losses with different prompt lengths are not the same metric.

### 8.2 Retention and forgetting

Let `R[t,b]` be read-only accuracy on block b after writing block t. Report its lower triangle and final row. For blocks already written:

```text
immediate[b] = R[b,b]
final[b] = R[T,b]
forgetting[b] = max_{u=b..T} R[u,b] - R[T,b]
```

Also report the direct immediate-to-final change because a historical maximum can reflect sampling noise. Plot recall by events/time since the last write, and score conflicting updates using the specified temporal semantics. Record before/after changes on untouched controls so apparent gains are not built entirely on lost original capability.

### 8.3 Runtime and statistics

For the main architecture, record retrieval frequency by VDB record, top-k weight entropy, per-token switching, retrieval consistency across paraphrases of one fact, and the norm ratio of injected residual to original projection output. LoRA/VeRA-bank controls also record slot load and normalized entropy, explicitly marking undefined max/min ratios caused by empty slots.

System metrics include peak GPU memory, separate authoritative CPU-VDB and GPU compute/transfer-buffer memory, full-request p50/p95 latency, per-token query/retrieval/modulation cost, actual transfer time, generation tokens/s, per-event write time, and storage bytes. Synchronize GPU timing boundaries, separate warm-up from formal examples, and report cold start, model loading, and steady-state inference independently. Current synchronous exact-CPU latency applies only to the measured scale, not large databases or ANN implementations.

Small single-seed runs serve exploration/software validation. Formal experiments need at least three seeds, with shared splits/event order across methods. Report each seed and means; bootstrap paired method differences by fact for synthetic tasks and by conversation/user for conversational tasks. Retain seed as a dimension of cross-seed inference rather than treating repeated questions under different seeds as independent large samples. Fix primary comparisons before testing; label exploratory ablations and multiple comparisons separately.

## 9. Draft preregistration for subsequent confirmation

These thresholds are **a plan to freeze before formal multiple-seed experiments**, not retrospective preregistration of ongoing or already observed pilot results. Dev difficulty may justify documented revisions. After formal testing, do not change primary metrics or thresholds in response to results.

| Hypothesis | Prespecified criterion | First response to failure |
| --- | --- | --- |
| H0: valid task/evaluation | Synthetic text-oracle EM reaches 90%; frozen no-memory backbone is clearly below oracle; read-only checksums stay unchanged | If even oracle fails, fix data, templates, decoding, or metrics before interpreting memory quality. |
| H1: VeRA parameter vectors carry information | On unseen entities, correct oracle values exceed empty/shuffled values by at least 10 percentage points, with paired-difference 95% CI lower bound above 0 | Check Wv/b gradients, support encoding, rank, and modulation before enlarging the VDB. |
| H2: usable actual addressing | In the formal main setting, real pre-answer recall@k ≥90%, and final EM is within 10 points of the same checkpoint's oracle | Improve Q/K alignment, granularity, and temperature; examine generation-time query changes. |
| H3: continual writing with frozen shared weights | Test facts improve clearly after VDB writing; at the prespecified longest stream, the first block loses at most 10 points from immediate to final accuracy, and untouched controls lose at most 2 points | Separate retrieval competition, conflict updates, value compression, and readout. Appending storage does not automatically solve forgetting. |
| H4: meaningful efficiency | Under a fixed GPU-memory or complete-read-latency budget, at least one method lies on the quality–cost Pareto frontier, counting shared projections, per-token CPU retrieval, and round-trip transfer | If only checkpoints shrink while reads become more expensive, report a storage trade-off rather than overall efficiency. |

90% and 10-point thresholds are project progression criteria, not consensus standards from the literature. Each stage needs difficulty-appropriate thresholds fixed during development. Complex real conversations should not be required to meet a synthetic text-oracle threshold.

## 10. Staged execution and student handoff

**Round one: run the main architecture and locate bottlenecks.** Fix one seed and a small synthetic stream. Run Frozen, Text oracle, and Text RAG first to check the task. Train shared b/Wq/Wk/Wv on independent training entities, then write input-derived keys/values for unseen entities. Compare actual within-layer per-token retrieval, oracle, empty, and shuffled values. Preserve negative results and report any gap between training-entity fitting and unseen-entity readout. LoRA-bank outcomes cannot substitute for the main method.

**Round two: freeze core comparisons for three seeds.** With three prespecified seeds and identical fact streams, compare the main method with static VeRA, VeRA/LoRA banks, text RAG, and an additive reader. Match or plot budgets separately for total trainable parameters, per-fact storage, and read/write time. Produce individual predictions, forgetting matrices, paired intervals, and cost tables. Select layer, rank, alignment weight, and learning rate only on independent dev data.

**Round three: transfer and mechanism ablations.** Add MedMCQA and LongMemEval development tracks. Prioritize insertion layer, r/dk, support pooling, top-k, contrastive/LM losses, tanh values/scaling, dynamic/static rank vectors, and per-token/question-cached queries. Large VDBs, CPU ANN, and GPU mirrors are separate system extensions. Change one major factor per round; enlarge long-term capacity only after both oracle readout and real addressing pass.

Each handoff-ready run contains configuration/hash, environment manifest, code commit, data-ID manifest, individual predictions/labels/retrieval information, training logs, optimizer-step count, memory/adapter checkpoints, aggregate metrics, costs, and reproduction commands. Base weights/environment caches can be reconstructed from revisions; trained artifacts, experimental data, and logs need independently verified backups. Public reports contain only verified results, excluding internal device information and private material.

The final report should answer a specific question, such as whether training-fact VeRA values are readable while values generated for new facts fail to generalize, or whether oracle works but per-token addressing is unstable. Even negative results of this form guide the next experiment better than training-loss reduction alone.
