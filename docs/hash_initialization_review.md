# Review of hash memory, initialization, and VeRA-Mem cold-start design

Review date: 2026-10-05. This document separates facts from original papers/public code from experimental recommendations for this project. It reports no new GPU results. New experiments are documented by their own configurations, logs, and run directories.

## Conclusions

Increasing training data is worthwhile, but the more important requirement is a stable interface among the writer, readable value representations, VeRA readout, and addressing on the same task. A nonempty, randomly initialized database provides capacity, not factual knowledge. A useful cold start should come from encoding legitimately observed training data, or from a trained parameter table with its matching reader. Engram/Qwen support joint training, normalization, and gating; they do not establish that filling a database with random values solves readout.

The project retains the user's main pipeline: **VeRA-layer inputs produce semantic queries → sparse vector-database lookup → retrieved values modulate VeRA → completed new observations are encoded as key/value records and written to the database**. Token hashing is suitable only as an independent addressing control or auxiliary candidate retriever. Replacing the main method with token-ID lookup would change the research question.

## Official model names and evidence boundaries

The user's reference to **Qwen3.8-Flash has an official counterpart**. Qwen's official model card describes it as a service version based on the open-source **Qwen3.8-Flash-Next**. This review concerns the open architecture, technical report, and public framework implementation; undisclosed service-side details are not treated as known. The main experiment still uses Qwen3-4B-Instruct-2507 as requested. [Official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [official Qwen repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## Engram: what is initialized and what is trained

Engram generates addresses from local input-token n-grams using deterministic multihead hashes; it does not learn a nearest-neighbor query-to-address mapping. After table-vector retrieval, a normalized inner product between the hidden state and projected key produces a sigmoid gate that controls linear-value injection. The table, projections, and backbone are jointly learned during pretraining. Main comparisons use 262B training tokens; smaller architectural ablations still use 100B. The appendix specifies Adam for table embeddings, a learning rate 5 times the base rate, weight decay 0, and zero-initialized short convolutions. It does not explicitly disclose the production table's initialization standard deviation or demonstrate training-free transfer to a frozen Qwen backbone. [Original Engram paper, §§2, 4, 6 and Appendix A](https://arxiv.org/html/2601.07372v1)

The official public release is a **demo** illustrating dataflow, not the complete training code. Evidence from fixed revision `fb7f84a21f91223715394a33a1dc24bbfb7f788e`:

| Item | Public-code observation | Interpretation |
| --- | --- | --- |
| Hash | Integer multiplication, XOR, and prime modulus fixed by seed/layer | Hash parameters are not learned through a semantic loss. This avoids the current learned-addressing co-drift, but does not supply semantic paraphrase generalization. |
| Table initialization | `MultiHeadEmbedding` directly constructs `nn.Embedding` without overriding initialization | The demo uses PyTorch's default distribution; this does not establish the production-training choice. |
| Key/value | Learnable linear layers; RMSNorm for Q/K | Gating is separated from the value representation; no `tanh` is applied to the value itself. |
| Gate | Normalized dot product divided by `sqrt(d)`, followed by signed square root and sigmoid | The demo adds a score transformation beyond paper equation (4); a reproduction must specify which version it uses. |
| Convolution | The demo constructs a convolution without explicitly zero-initializing it | The appendix specifies zero initialization, further showing that the demo is not a complete training recipe. |

[Fixed-version hash code](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L188), [table and readout](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L305), [convolution](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L123)

PyTorch documents `N(0,1)` as the default weight distribution for ordinary `nn.Embedding`. This describes the demo's construction, not a recommended scale for current VeRA-Mem. Large numbers of untrained random slots do not supply semantic knowledge. [PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

## Qwen3.8-Flash-Next: the verifiable initialization path

The technical report uses multihead n-gram hashing and contextual gating; these ablations use 300 training tokens per active parameter. Larger tables reduce training loss, but downstream metrics do not improve monotonically. Table optimization uses Adam without weight decay. A separate **QSA attention indexer** is warmed up by distillation and then jointly trained. That procedure can inspire our curriculum, but must not be described as the n-gram table's initialization algorithm. [Qwen technical report, §§2.1.2, 2.3, 3.1](https://arxiv.org/html/2608.30320v1)

The official Qwen repository primarily supplies documentation and the report; the public model implementation is in Transformers' `qwen4_exp`. This review read fixed commit `f5ab85619d989359ef47b5efed8a91a15045627b`, with official model configuration revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

| Item | Verified path | Conclusion |
| --- | --- | --- |
| Hash table | `Qwen4ExpTextNGramEmbedding` constructs `nn.Embedding` | Fixed integer hashes retrieve learned rows; this is not ANN/VDB semantic nearest-neighbor lookup. |
| Table/linear initialization | The model's `_init_weights` calls `PreTrainedModel._init_weights`; the parent initializes embeddings/linear layers with a zero-mean Gaussian; the official config sets `initializer_range=0.02` | **Constructing a fresh model from the public configuration** initializes the table as `N(0,0.02²)`. Loading a pretrained checkpoint uses saved weights. This does not establish identical initialization at every internal production-training stage. |
| Readout | Linear value projection with a normalized Q/K gate | Like the Engram demo, it applies a signed-square-root score transform; stored values are not represented as `tanh(value)`. |
| Convolution/normalization | PLE convolutions are explicitly zero-initialized; RMSNorm weights are zero, with `1 + weight` used in the forward pass | Effective normalization gain starts at 1; this must not be interpreted as zeroing the whole memory branch. |

[Table construction](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1072), [gated readout](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1175), [model initialization override](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1325), [parent initialization](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/modeling_utils.py#L2297), [official configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

## Testable adaptations for VeRA-Mem

The following are designs and hypotheses proposed for this project, not measured conclusions of the papers above.

### 1. Separate value stability from memory-relevance gating

The old `tanh(Wv h)` simultaneously limits amplitude and encodes content. Once most coordinates approach ±1, gradients shrink and different observations may map to identical sign vectors. First compare:

- The old `tanh`, retained as a failed baseline.
- Linear values after training-statistics centering/normalization, with vector-level RMS limits if necessary instead of coordinate-wise clipping.
- The same values plus a separate memory gate controlling injection strength, conditioned on agreement between the query and retrieved key.

Do not initialize output gain, value encoder, and gate all to exact zero. At least one learnable gradient path must remain. Ablate a small nonzero output scale separately from exact zero-residual initialization. Setting bounds and monitoring gradients is better grounded than copying `std=0.02`. Values may have controlled norms, but variance, mean cosine, effective rank, and per-coordinate gradients must be recorded so that zero saturation does not conceal identical vectors.

Fixed random VeRA A/B and shared output scaling may still restrict the readability of arbitrary values. First retain the architecture while testing rank 64/256 and reader-training data scale; compare oracle values and random permutations under matched conditions. If the oracle still does not learn, measure whether values retain answer information and whether `B diag(value) A x` provides enough controllable directions. A learned low-rank B or added value-residual readout should be an explicitly labeled expressivity control, not an unannounced replacement of the requested VeRA method.

### 2. Establish the read/write interface before joint addressing training

A three-stage approach is proposed:

1. **Address warm-up:** use question/observation pairs from training entities to train or initialize Q/K, with independent dev entities/templates. Record Recall@1/@k and positive–negative margins.
2. **Reader warm-up:** temporarily freeze address encoders and train writer/VeRA with the correct value. Test the oracle on unseen entities to rule out simple memorization of training questions.
3. **Sparse joint training:** restore real top-k while retaining an auxiliary address loss. Use a lower learning rate for Q/K than for writer/reader. Distillation toward the warm-up retrieval distribution or parameter anchoring can be considered. Compare frozen versus fine-tuned Q/K before deciding which to use for continual writing.

Changing the key encoder requires re-encoding or versioning old keys; stored representations and current queries must not silently become incompatible. Rebuild offline banks from training supports each round. Initially freeze writer/Q/K online and permit only incremental writes, preserving causality and interpretability. Any later online SGD must be a separate condition with write time, forgetting, and supervision accounted for.

Training alignment only at the final question token while querying every inference token creates a position mismatch. In addition to first-token Recall@k, record correct-record residency during generation, candidate switching, positive–negative margins, and read entropy. High Recall@4 does not mean the softmax assigns enough weight to the correct value. Select top-1/top-4, temperature, and empty-read gates on dev data. Question-level fixed retrieval can diagnose this issue but remains an ablation, not a replacement for the main per-token query implementation.

### 3. Distinguish at least four cold-start conditions

| Condition | Initial bank | Research question |
| --- | --- | --- |
| Empty | No records | Online writing from an empty bank |
| Random | Random vectors matched in count and bytes to cold start | Whether nonemptiness itself helps; also tests noise sensitivity |
| Train-prefill | Keys/values encoded only from training supports, or table rows fitted on training data | Whether the learned distribution, long-term knowledge, and gate help a new stream; future test facts are excluded |
| Observed-prefill | Background observations legitimately supplied before the stream starts | Continued learning with prior knowledge; background inputs count toward system resources and the information budget |

For a new-random-entity → new-random-answer task, train-prefill does not contain new test facts. The system should not be expected to predict a hidden random answer on first encounter. Its value is measured after legitimate writing, through readout, interference resistance, and stability. Placing all test answers in the bank beforehand is a valid rereading/recall experiment, but not pre-write generalization or a fair cold start.

A stronger training-table initialization could use `v_i = encoder(support_i) + learned_residual_i`, optimize only training rows and the reader, then distill row residuals back into the writer. Online values for new facts would still be writer-generated. Test unseen entities and unseen answer combinations together to prevent a system that can read only directly optimized old rows. Include row parameters, optimizer state, and offline training cost in comparisons.

### 4. Separate data growth from compute growth

Use nested sets of 128/512/2048/8192 training facts, with disjoint dev/online-test entities, and at least three training seeds for final candidates. First compare data diversity at fixed optimization steps; then compare fixed epochs when additional compute is allowed, reporting total tokens, updates, and GPU time. Initial runs can cover 512/2048 before deciding whether dev curves justify more; no particular sample count is guaranteed to cross the readout threshold.

Each training fact should have multiple query/support templates, irrelevant padding, conflicting updates, and multitoken answers. Answers should be random rather than derived from entity IDs. Establish readout on a diagnostic finite answer vocabulary before testing answer combinations, paraphrases, and real conversations. Template changes must be shared across methods; an easier task for the new method cannot be compared directly against a harder old task.

Evidence of completion should include real retrieval beating empty reads and permuted values, a narrower oracle–real gap, success on new entities/paraphrases, preserved old facts and conflict updates, and complete capacity, latency, norm/rank, and gradient diagnostics. Lower training NLL, a nonempty bank, or nonidentical values alone do not establish successful parameterized-memory learning.

## Boundaries to retain in the student handoff

- Random hash-table initialization, loading a pretrained table, and constructing a VDB from legitimate observations are three different cold starts.
- Engram/Qwen initialization and optimization settings come from different architectures and scales. They are reference points, not evidence of effectiveness here.
- These static trainable tables primarily learn through offline gradients; they are not automatically continual-learning systems that append observations during inference.
- The proposed nonsaturating values, training-bank warm start, and curriculum-based joint training require new project experiments. Preserve negative results to distinguish capacity, addressing, readout, and data-scale limitations.
