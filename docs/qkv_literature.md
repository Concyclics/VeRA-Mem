# QKV, associative memory, and sparse VDBs: literature for the next design

Checked on 2026-10-06. This document presents literature and experimental suggestions. **It is not a frozen experimental protocol and does not claim that the proposed changes have worked.** The eight papers were read through their full `.md` texts using the Hugging Face papers skill, with mechanisms checked against arXiv and authors' repositories. When the browser could not render a markdown response, the same public endpoint was downloaded directly. No third-party review or HF-generated summary was used as evidence.

## The hypotheses most worth testing first

The user's observation that the current system resembles “selecting existing vectors” reflects the results: the model can switch between seen A/B answers, but writing a novel combination does not produce the correct output. However, ordinary attention also combines existing values. **Combining existing vectors and attention-style generalization are compatible.** The important requirements are transferable Q/K addressing, decodable content in V, and training that forces the model to use the current episode's binding rather than a fixed entity–label mapping in shared parameters. [Attention §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2) and [Fast Weight Programmers §6.1](https://arxiv.org/html/2102.11174v3#S6.SS1) support this distinction.

The code already has Wq/Wk/Wv, a query derived from the actual VeRA layer input, softmax within top-k, and continuous value mixing. It would be incorrect to say that earlier models lacked QKV or performed only discrete label selection. The bottleneck has not been causally identified: in the [previous report](reconstruction_results.md), three-slot C/D conditions had high fact R@4, but the third-word slot was often absent from the read. Even when that slot was retrieved at some point in the trajectory, the full answer remained wrong. Addressing, read timing, value encoding, and multiplicative readout are all still candidate causes.

The two priorities are therefore **random rebinding in each episode**, to break fixed entity–old-label associations, and **retrieving a fact group before querying its content slots**, so that hitting its first two words is not mistaken for reading the complete fact. Random associative-recall experiments and LongMem provide direct precedents respectively, but their transfer to VDB→VeRA remains our experimental hypothesis.

## Eight directly relevant primary sources

### 1. Attention Is All You Need (2017)

**Paper evidence.** Section 3.2.1 outputs `softmax(QKᵀ/√d) V`; §3.2.2 concatenates multiple independently projected heads and applies an output projection; §3.2.3 uses decoder queries and encoder K/V for cross-attention. A single head's softmax mixture lies in the convex hull of the retrieved values; subsequent output projection, residual connections, and nonlinearities provide further transformations. [Paper §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)

**Design implication and limits.** Query and storage may come from different feature domains. Prefix-dependent queries, retained position/role information, and multiple heads have established precedents. The paper does not guarantee that deeper QKV networks solve our compositional failure, nor does it evaluate an external CPU VDB, per-record replacement, or low-rank multiplicative VeRA. Additive `W_O read(q)` is an attention-path diagnostic, not evidence that the user's original design has succeeded.

### 2. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (2020)

**Paper evidence.** Sections 3.2–3.4 rewrite kernelized attention as an accumulated associative state `S=Σφ(k)vᵀ`, with a separate normalization state. Queries read from that state; causal updates depend only on the observed prefix. For a fixed feature dimension, inference-state size does not grow with history length. Experiments include synthetic tasks, image generation, and speech recognition. [Paper](https://arxiv.org/abs/2006.16236)

**Design implication and limits.** Temporary parametric memory need not select static embeddings; a forward pass over new content can construct key–value associations. However, superposing records in a fixed matrix differs from an individually replaceable, auditable VDB, and a finite-dimensional kernel is not automatically equivalent to softmax attention. It is initially useful as an associative read/write positive control without retrieval approximation, not a drop-in replacement that can be claimed to preserve the main design's editing semantics.

### 3. Linear Transformers Are Secretly Fast Weight Programmers (2021)

**Paper evidence.** Section 3 connects linear attention with fast-weight outer-product writes; §4.2 builds an error-correcting update from the previously retrieved value; §6.1 randomly samples K/V for each sequence and presents the query only at the end. Section 6.1.2 also permits repeated writes to the same key and requires its latest value. Synthetic values are fixed one-hot vectors, with a regression loss. [Paper §4.2](https://arxiv.org/html/2102.11174v3#S4.SS2), [§6.1](https://arxiv.org/html/2102.11174v3#S6.SS1), [authors' code](https://github.com/ischlag/fast-weight-transformers)

**Design implication and limits.** This is the most direct precedent for training random bindings and testing overwrites: training learns a rule for using memory, while a new episode determines test bindings. It does not show that a frozen LLM can decode three-word content; one-hot regression is also different from free natural-language generation. The established abbreviation is **FWP**; this document uses it rather than treating “TFW” as another verified model name.

### 4. Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet, 2024)

**Paper evidence.** The write in §2.2 can be expressed as `S_t=S_{t-1}+β_t(v_t−S_{t-1}k_t)k_tᵀ`: read the old value for the current key, then write the error. Section 3 provides a parallel training algorithm; §4.1 includes synthetic tests such as MQAR. Language-model experiments train 1.3B parameters on 100B tokens and compare hybrids with local/global attention. [Paper](https://arxiv.org/html/2406.06484v3), [authors' FLA implementation](https://github.com/fla-org/flash-linear-attention)

**Design implication and limits.** In compressed associative state, addition differs from error-based replacement; conflicting keys and residual old values are useful tests. The current VDB already supports exact single-vector upserts and does not require a delta rule to physically remove an old value. Current errors do not imply that the database failed to update. A fast-weight matrix would be a different storage mechanism whose collisions, capacity, and write order must be recorded. Large-scale LM results do not guarantee success for a small VeRA interface.

### 5. Gated Delta Networks: Improving Mamba 2 with Delta Rule (2024/2025)

**Paper evidence.** Section 3.1 combines a decay gate with the delta update. Section 3.2 distinguishes retention from filtering: decay may hurt long-term retention of a simple needle while helping remove substantial realistic interference. Section 3.4 uses linear projections, short convolutions, and SiLU for Q/K/V, plus L2 normalization for Q/K. Section 4 compares 1.3B parameters trained on 100B FineWeb-Edu tokens. [Paper §§3–4](https://arxiv.org/html/2412.06464v1), [authors' FLA implementation](https://github.com/fla-org/flash-linear-attention)

**Design implication and limits.** QKV contextualization and write/forget gates should be ablated separately; the paper does not support “adding a gate always improves memory.” It operates on finite recurrent state over a sequence, not independent records in a persistent VDB. Sparse external memory can borrow normalization, content-dependent queries, and update-conflict tests without attributing the full GatedDeltaNet improvement to replacing Wq alone.

### 6. Memorizing Transformers (2022)

**Paper evidence.** Section 3.1 uses the current query for external kNN lookup at one layer, recomputes softmax attention over retrieved K/V, and combines it with local attention using a head-wise gate. Historical K/V are non-differentiable and document-specific. Section 3.2 discusses training-induced cache staleness and Q/K normalization. The main experiments in §4.2 train from scratch for 500 k steps with 2¹⁷ tokens per step and k=32; §4.5 also adapts an existing model to external memory. [Paper](https://arxiv.org/abs/2203.08913), [authors' Meliad code](https://github.com/google-research/meliad)

**Design implication and limits.** Non-differentiable retrieval can work with a trainable attention reader; sparse retrieval does not imply fixed-label selection. However, values enter attention activations rather than serving as VeRA parameters. Language-modeling gains are not evidence of successful individual-fact edits, and the paper does not establish that our tiny training budget is sufficient for the same abilities.

### 7. Augmenting Language Models with Long-Term Memory (LongMem, 2023)

**Paper evidence.** Section 2 encodes K/V with a frozen backbone and queries/fuses them with a trainable residual SideNet, reducing cache staleness caused by encoder updates. Section 2.3 selects chunks using mean keys, then expands their token K/V for attention. Section 3.1 uses a 407M backbone, a 12-layer SideNet, 26B adaptation tokens, a 65 k-token bank, chunk length 4, and 16 retrieved chunks totaling 64 token K/V pairs. [Paper §2.3](https://arxiv.org/html/2306.07174#S2.SS3), [§3.1](https://arxiv.org/html/2306.07174#S3.SS1), [code entry supplied by the paper](https://aka.ms/LongMem)

**Design implication and limits.** This directly precedes “fact-level indexing plus content-slot reads.” Frozen source encoding and a trained reader also support cache reuse. If Wk/Wv are still trained, projected caches can still become stale and must be re-encoded or versioned. The complete SideNet has far more freedom than a single-layer rank 64 interface, so its performance should not be assumed. Group retrieval also changes the number of returned vectors; bytes and readout cost must be reported separately.

### 8. Zoology: Measuring and Improving Recall in Efficient Language Models (2023/2024)

**Paper evidence.** Section 3.2 extends single-query associative recall to MQAR with multiple queries and varying distances; §4 analyzes recall in different mixers; §5 studies input-dependent sparse interactions. The paper notes that some older synthetic tasks are easy to pass yet fail to explain recall gaps in real language. The authors' repository provides small synthetic architecture tests. [Paper](https://arxiv.org/html/2312.04927), [authors' code](https://github.com/HazyResearch/zoology)

**Design implication and limits.** Tests should vary position, query, and novel binding rather than use only a fixed template or first-prediction hit. Independent episodes can associate the same entity with different contents, weakening fixed-mapping shortcuts. This specific VeRA training design is our hypothesis, not a result established by the paper. Token association in MQAR also cannot replace tests of natural-language semantics, ordered three-word reconstruction, or locality under continual editing.

## Keep the main mechanism distinct from diagnostics

In column-vector notation, let `h_t` be the actual current VeRA input and `q_t=f_Q(h_t)`. Each VDB record `(k_i,v_i)` is constructed from observed text. Retrieve a set `I_t`, then read `m_t=Σ_{i∈I_t} softmax(score(q_t,k_i)) v_i`. This is conceptual shorthand; normalization, scaling, and gates must follow the new implementation and protocol.

| Path | Use of the retrieved memory | Relation to the user's objective | Question it can answer |
| --- | --- | --- | --- |
| Dynamic VeRA main path | `Δh_t = diag(b) B diag(m_t) A h_t` (schematic) | Retains VDB-vector modulation of a parametric increment | Can new content control inference through the same multiplicative interface? |
| Grouped attention→VeRA | Sparsely select a fact group, query its content slots to obtain `m_t`, then modulate as above | Retains the main path while changing QKV/reading | Does finer addressing and content reading improve writability? |
| Additive attention diagnostic | `Δh_t = W_O m_t`, with other conditions matched | Changes injection; it is not VeRA success | Success here with failure on the main path would motivate further isolation of multiplicative-interface limitations |
| Fast-weight / delta state | Write `S`, read with `S q_t`, then choose an injection mechanism | A different storage/update mechanism | Association learning and overwrite capacity; VDB single-record deletion/restoration guarantees do not transfer automatically |

Additive/multiplicative comparisons must report output magnitude, trainable parameters, projection initialization, and optimization budget, because differences may also arise from scale or capacity. Failure of both does not directly show that a VDB cannot generalize; success of both still requires checking dependence on the current values rather than shared labels.

## Minimal experiments that can distinguish mechanisms

These are suggestions only. Formal parameters, data seeds, and thresholds must be sealed separately; rules cannot be added after inspecting new confirmation results.

1. **Train new bindings before expanding the corpus.** With a small vocabulary and bank, randomly generate entity–three-word bindings per episode so that the same entity sees multiple complete payloads during training. Seal training, development, and confirmation combinations and bindings separately; use development for architecture decisions and confirmation only for verification. Match fixed-binding and episodic controls for updates, target exposure, and sequence length. Randomly rebound support is legitimate write material; the student prompt still contains only the question.
2. **Separate addressing changes from content reading.** Start from a fixed main-path baseline and compare whole-fact retrieval followed by within-group attention. Address keys may come from the entity/relation span in the support, while values come from the content span. Entity text is legitimately observed; inserting an array fact ID directly into queries or search grants additional oracle access. Slot positions/roles may come from observed text, but the gold next word or correct answer position must not choose the slot. Do not change multi-head attention, nonlinear QKV, and larger top-k all at once.
3. **Hold bank size fixed and separate novel combinations from new entities.** First test new bindings/combinations with the same 16-record bank, then replace entities while keeping bank size fixed. Adding 64/256 distractors is a separate axis. This avoids the previous round's simultaneous changes in entity, payload, and bank size.
4. **Compare two readouts with the same retrieval and values.** Keep multiplicative VeRA as the main path and label additive attention as a diagnostic. Record each generated token ID, query, and target group/slot retrieval and weights; distinguish whether the third-word slot was used at the relevant prediction from whether it appeared anywhere in the trajectory. Gold-prefix NLL or R@k is not full-answer success.
5. **Test whether the current binding overrides the old one.** Freeze parameters and independently update each target A→new B→A. Include value shuffle, empty memory, correct-key/wrong-value controls, and a non-target neighbor. Report novel-content EM, strict update+restore, single-world real-control differences, and correct neighbor preservation. Complete old-A/B outputs are descriptive diagnostics, not a basis for filtering confirmation cases.
6. **Validate the teacher and cache contract first.** Give the teacher the raw support, the same question, and the same generation budget; use it only for task qualification. Rebinding by swapping old contextual payload hidden states may retain the old entity or prefix and is not re-encoding a new note. Either re-encode the actual rebound text with the frozen backbone or label the experiment explicitly as a cached-feature-binding diagnostic. Freezing the backbone does not make projected K/V permanently immune to staleness.

The desirable outcome is not a higher attention score: with shared parameters frozen and novel bindings sealed, complete generated content should change correctly after an individual VDB update, recover after restoration, and preserve other facts. This standard still requires experimental verification. The papers support trying the direction, but neither predict inevitable success nor justify replacing mechanism validation with a larger corpus and a total-token count.
