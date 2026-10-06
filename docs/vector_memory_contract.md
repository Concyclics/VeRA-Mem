# VeRA-Mem vector-parameter memory contract and experiment plan

Status: **plan and implementation constraints, 2026-10-05**. This page defines the system to implement and verify; it does not claim that planned training or controls have completed. Executed configurations, training logs, checkpoints, individual predictions, and result reports establish completion. Implementation changes must be reflected here while retaining old configurations.

## User requirements and boundaries of the main method

The main method must preserve this loop:

1. At inference, generate a query vector from the actual input tensor of the layer containing VeRA.
2. Use that query to sparsely retrieve a set of value vectors from the VDB.
3. Feed retrieved vectors into VeRA's parameterized branch to change subsequent neural inference.
4. Encode new inputs/revealed observations with a writer into keys/values and write them to the VDB so later inference can use the new knowledge.

Queries must not be replaced by external text IDs, known correct-evidence IDs, or answer labels. The main branch neither concatenates retrieved text into the prompt nor substitutes another answer classifier for the backbone's output. Hash methods motivate initialization, numerical stability, and training. Hash controls, text RAG, oracle readout, and fixed-query ablations must be labeled as controls rather than substitutes for semantic vector retrieval. See the [hash-initialization review](hash_initialization_review.md).

## Tensors and parameters

Use Qwen3-4B-Instruct-2507 at a fixed revision. The current insertion point is a layer's `mlp.down_proj`; `x_t` is that linear layer's actual input, not a final-layer hidden state or another model's embedding. Save the layer index, dimensions, rank, and every training setting in the run configuration.

Let the base linear layer be `W ∈ R[d_out × d_in]`, with original output `W x_t`. VeRA's fixed random projections are `A ∈ R[r × d_in]` and `B ∈ R[d_out × r]`; shared output scaling is `b ∈ R[d_out]`. The database stores:

```text
M_s = {(record_id_i, timestamp_i, key_i ∈ R[d_k], value_i ∈ R[r])}
```

The main read forward pass is:

```text
q_t       = L2Normalize(Wq · normalize_query(x_t))
scores_i  = q_tᵀ key_i                       # key is already L2-normalized
I_t       = TopK(scores, k)
alpha_t   = softmax(scores[I_t] / temperature)
vbar_t    = Σ(i∈I_t) alpha_ti · value_i
delta_t   = diag(b) · B · diag(vbar_t) · A · x_t
y_t       = W · x_t + delta_t
```

Conditioned on the current input and read result:

```text
W_effective(x_t, M_s) = W + diag(b) B diag(vbar_t) A
```

This is the meaning of vectors as parameterized memory: values act as input-dependent rank-dimensional parameter scales driving the VeRA branch. It does not mean the base `W` has been rewritten, or that every VDB record is an independently SGD-optimized parameter. Because `vbar_t` depends on `x_t`, the complete module is an input-dependent nonlinear mapping, not one fixed low-rank matrix.

For an empty bank, `vbar_t=0` and the memory residual is zero. The normal main method forms a query from the actual layer input at every valid prompt/decode token. One fixed retrieval per question is only an explicitly labeled ablation. A response uses one database snapshot, but different tokens may retrieve different records.

The current CPU reference VDB performs an exact cosine scan: scoring costs `O(N d_k)`, followed by mixing only top-k values. Sparse value access does not make query complexity independent of bank size and is not Engram's deterministic `O(1)` hash lookup. Scaling experiments must report actual retrieval/transfer costs.

## Writer and causal timing

A new `observation_s` is independently encoded once by the frozen backbone to obtain feature `h_s` at the same insertion layer. This round uses the specified support representation and selected token. Distinguish that choice from query positions and teacher-forcing label positions.

```text
key_s   = L2Normalize(Wk · normalize_support(h_s))
value_s = ValueEncoder(h_s)
M_(s+1) = upsert(M_s, record_id_s, timestamp_s, key_s, value_s)
```

An online event proceeds as follows:

1. Answer using existing `M_s` and record the pre-write prediction. The new fact's answer is not yet accessible to the query, cold-start bank, or retrieval filter.
2. The stream legitimately reveals the complete observation, such as a new fact, user correction, or completed-task feedback.
3. Disable memory injection, extract frozen-backbone features, and generate a key/value with the trained writer. The writer may now encode the newly revealed answer because it is observed input.
4. Perform an atomic upsert. Update existing record IDs by timestamp and append new IDs. Do not retain duplicate positive records for the same fact in one snapshot to inflate top-k weight.
5. Subsequent queries or explicitly labeled post-write retests use `M_(s+1)`. Read-only evaluation must not silently write, replay, or modify the database.

Record IDs support persistence, updates, and evaluation audit. Main `search` accepts only a query and must not filter candidates by the correct ID. Conflict updates rely on legitimately supplied record identity. Upserting a known ID does not establish open-text entity disambiguation or contradiction detection.

Existing memory is disabled while encoding observations to prevent old answers from feeding back into new keys/values. Future multilayer adaptation, backbone updates, or online writer updates require a new compatibility check for old key/value versions. Records from incompatible encoders must not be mixed.

## Offline training, online writing, and TTT

| Object/stage | Offline interface training | Online inference/writing in this round |
| --- | --- | --- |
| Qwen backbone `W` | Frozen, while retaining backpropagation from later layers to the module | Frozen |
| Fixed random `A/B` | Not optimized | Not optimized |
| `Wq/Wk/Wv/b` | Trained according to the curriculum; exact freeze states are configured | All frozen |
| Normalization centers | Fitted only from training query/support features | Fixed; no dev/test/future-observation statistics are incorporated |
| Initial training bank | Recomputed from legitimate training supports with the current writer; selected values/scores retain gradients to the shared interface | An ordinary key/value snapshot generated and saved by the final writer |
| New online records | Not applicable | Forward encoding and upsert of legitimate observations; no optimizer step |

The offline bank is not a set of free `nn.Parameter` slots independent of the writer. Hard top-k indices are nondifferentiable, but selected scores and values can retain their graphs. The persistent CPU query interface detaches tensors and must not mistakenly be used as the entire offline gradient path.

This round **learns the read/write interface offline and modifies conditional parameter memory through online forward-pass writes**. It supports incremental storage, but is not TTT-style inference-time self-supervised gradient updating of fast weights, nor online fine-tuning of Qwen. Later online SGD, delta-rule fast weights, or independently trainable slots must be separate conditions with explicit learning signals, state, gradients, and write budgets.

## Planned stable variant

The stable variant retains the VeRA branch, replacing the coordinate-wise tanh writer with training-domain centers and RMS-normalized linear values:

```text
u(x)       = x / RMS(x)
z_q(x)     = RMSNormalize(u(x) - mean_train_query)
z_s(h)     = RMSNormalize(u(h) - mean_train_support)
q(x)       = L2Normalize(Wq z_q(x))
k(h)       = L2Normalize(Wk z_s(h))
v(h)       = RMSNormalize(Wv z_s(h))
```

All denominators include the configured epsilon. Overall value RMS is controlled, but individual coordinates need not lie in `[-1,1]`. The old `abs(value)>0.99` saturation statistic therefore does not apply directly to stable values. For both versions report per-coordinate variance, effective rank, intersample cosine, output norm, and gradients; additionally report near-zero derivatives specifically for tanh.

The raw/stable comparison changes query/support centering, value nonlinearity, learning rates, retrieval temperature, and curriculum together. It is a **combined-training-recipe comparison**. Differences cannot be attributed solely to removing tanh. A single-factor causal conclusion requires later controlled ablations. Data-scale comparisons within a variant retain its recipe.

## Planned experiment matrix

First run `stable: train=32, updates=128` to diagnose the pipeline and learning: model, actual-layer hook, losses, gradients, database I/O, and save/restore. It includes 100 address warm-up updates, then 64 oracle and 64 real-top-k updates to exercise prediction-token addressing and real-retrieval gradients. It is not guaranteed to fit the training set, and successful fitting is not assumed. Online evaluation uses entities from independent seed `8042`, with 16 stream facts and 16 controls, preserving main-matrix test stream seed `7042`. This run does not enter main-matrix superiority conclusions.

The proposed main comparison follows. Names are design labels; saved configurations determine formal run IDs.

| Condition | Training entities | Module | LM batch | LM updates | LM target exposures |
| --- | ---: | --- | ---: | ---: | ---: |
| raw-staged-small | 128 | tanh, value-only centering; oracle reader throughout | 8 | 512 | 4096 |
| raw-staged-large | 4096 | Same as above | 8 | 512 | 4096 |
| stable-small | 128 | Centering/RMS values | 8 | 512 | 4096 |
| stable-medium | 1024 | Centering/RMS values | 8 | 512 | 4096 |
| stable-large | 4096 | Centering/RMS values | 8 | 512 | 4096 |
| stable-large-extended | 4096 | Same as stable-large; separate training-budget control | 8 | 1536 | 12288 |

The first five conditions use nested training entities and reproducible without-replacement target traversals, cycling as needed: 128 records repeat 32 times, 1024 repeat 4 times, and 4096 run once. This compares diversity at equal target exposures, not equal total FLOPs or wall time. Separately account for feature construction, alignment, bank encoding, and validation.

The additionally prespecified `extended` profile keeps 4096 training entities and increases LM updates from 512 to 1536: three traversals of the same pool and 12,288 target exposures. It retains 400 address warm-up updates, 128 test facts, an empty initial bank, and the remaining recipe. Oracle/real phases have 768/768 updates, each half of the run. It tests **additional training budget**, a different question from increasing sample count. It is a complete, independently initialized run, not a continuation selected from the shorter run's best checkpoint. 3× refers to LM update budget, not necessarily 3× end-to-end cost including fixed preprocessing/address warm-up.

The main matrix, extended profile, and cold-start controls share 128 new test entities and 64 controls. Controls are unknown entities with hidden random answers, not automatically an abstention evaluation. Dev entities must be independent of train/test, and checkpoints/thresholds must be selected only from train/dev. If test results inform changes, later confirmation must use new test entities rather than repeatedly treating test as dev.

The configured curriculum is as follows; logs still determine whether execution completed:

1. Address warm-up: 400 updates, minibatch 128. Full batches give **51,200** address-pair exposures, reported separately from 4,096 LM target exposures. Small-data runs may see all 128 records each step; large-data runs average 12.5 repetitions. These different data-coverage conditions must be disclosed.
2. Both raw conditions retain the staged baseline: Q/K update learning rates are 0 after warm-up, and all 512 LM updates use an oracle reader. Real evaluation uses `top_k=4`, `temperature=0.2`. This tests whether scaling data helps the original recipe at equal exposures.
3. The three stable conditions use 256 oracle-reader updates to establish writer/VeRA readout, then 256 actual sparse-read updates; `top_k=4`, `temperature=0.05`. The real stage applies auxiliary addressing losses at question and answer-prediction positions, jointly optimizing Q/K at smaller learning rates. Source snapshots/configurations specify exact rates. Neither the oracle curriculum nor oracle diagnostic scores replace results from the real-retrieval main method.

Initial-bank controls are **0 versus 128 train-only bootstrap records**. The fixed 128-record background comes from training observations, excluding test entities/future answers. Real-read offline episodes include the same background and recompute keys/values with the current writer so the reader learns with the initialization distribution. The oracle stage is only an ideal-evidence curriculum condition: even with a background bank present, forced correct values are not retrieval under real background interference.

Deduplicate background/current supports by fact ID within each episode. Cold-start training uses a fixed pool of 4096 facts, with 128 background records covering about 3.125%; do not mix this with adding bank128 to train128. Still record whether each target belongs to the initial bank and stratify when needed, so additional exposure from repeated background is not hidden.

Use a **2×2 cross-evaluation** of training background and inference initialization:

| Fixed background during training | Inference initial bank=0 | Inference initial bank=128 |
| --- | --- | --- |
| 0 | `stable4096`, train and evaluate | `standardtrained_coldinit`, load the same `stable4096/best.pt`, evaluate only |
| 128 | `coldtrained_emptyinit`, load the same `stable4096cold128/best.pt`, evaluate only | `stable4096cold128`, train and evaluate |

Each row shares exactly the same checkpoint with no extra training steps, isolating inference initialization. Columns compare recipes that did or did not encounter fixed backgrounds offline. Both trainings retain train size, updates, seed, and dev-selection rules. Evaluation-only runs must record training provenance and evaluation `cold_bank_size`, without overwriting the checkpoint's original training condition or reselecting a checkpoint on test data.

Every online condition processes legitimate new observations one by one, writes keys/values, and records pre-write, post-write, delayed recall, paraphrase, and control outcomes. Retain real retrieval, oracle, value permutation, and zero-residual controls. Confirm final configurations with multiple seeds; a single-seed main experiment supports only preliminary selection.

## Completion and interpretation checks

| Claim to establish | Required evidence |
| --- | --- |
| Actual layer inputs drive retrieval | Hook inputs match actual queries; prompt/decode use the main forward path; no correct-ID routing |
| Values drive the parameterized branch | Paired predictions for correct, permuted, and zero values; dynamic-branch formula and shape checks |
| New online facts enter memory | Before-read/after-write content hashes, record IDs, timestamps, and new keys/values; evaluation does not mutate the bank |
| No future-label leakage | Train/dev/test ID/entity intersections; initial-bank provenance; event-order and feature-input audits |
| Scaling helps | Complete equal-target-exposure matrix; independent dev selection; paired shared-test results; multiple-seed confirmation and cost accounting |
| Stability improves | Variance/rank/norms, gradients, positive–negative margins, retrieval distributions; separate tanh-specific saturation metrics |
| Addressing improves | Recall@1/@4, correct-value softmax mass, generation residency/switching; top-4 hits alone are insufficient |
| Readout works | Oracle beats permuted/zero values on unseen entities; real retrieval approaches oracle; paraphrases do not fail completely |
| Cold start helps | Legitimate information budget and added bytes for train-only banks; empty/warm controls, old-fact retention, new-fact learning |
| Reproducibility and handoff | Complete configuration, model/source revisions, seeds, logs, checkpoints, individual predictions, and verified backups |

Results may independently support data-scaling benefits, improved numerical behavior of the combined stable recipe, or better/worse interference with an initialized bank. Report negative outcomes honestly. Lower training loss or successful database writes alone do not establish useful continual learning.
