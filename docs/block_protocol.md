# Sparse Block Memory and Dynamic Low-Rank Operators: Experimental Protocol

> English translation of the historical protocol. This localization is not a new preregistration and does not replace the sealed source or its hashes; see [reproducibility.md](reproducibility.md).

**V 1 preregistered protocol (sealed before formal training), 2026-10-06.** This document specifies formulas, data, budgets, metrics, and criteria for further research. Before formal training, the coordinating agent records the source, data, plan, and document SHAs. At registration, formal results had not yet been produced. After sealing, C/D scores must not change the formulas, templates, training steps, or thresholds. See [block_literature.md](block_literature.md) for the literature and equivalence arguments, and [qkv_results.md](qkv_results.md) for the preceding negative evidence.

The question is: **after sparsely retrieving a complete three-slot value matrix, does generating a dynamic low-rank operator separately from each slot improve retrieval of untrained complete content combinations compared with pooling first and then generating an operator?** Moving a shared linear projection before summation is mathematically equivalent and does not constitute a new model. This round retains the original diagonal VeRA path and compares two additional operators using the same actual grouped retrieval.

## 1. Fixed matrix and formulas

| Training bindings | diagonal | pooled_outer | block_outer |
| --- | --- | --- | --- |
| static: fixed A/B for each entity | static_diagonal | static_pooled_outer | static_block_outer |
| rebind: reassigned every epoch | rebind_diagonal | rebind_pooled_outer | rebind_block_outer |

Each of the six arms uses seeds `91042 / 91043 / 91044`, giving **18 models**. Shared tensors have bitwise-identical initialization within each seed; the new tensors in both outer arms are also identical. The model is Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The layer-20 down projection has input/output dimensions 9,728/2,560; latent rank and key dimension are both 64. The backbone and A are frozen; B is trainable. Foundation memory is disabled. The three contextual word slots, shared Wq/Wk/Wv, and zero-initialized slot_position reuse the validated interface.

At every actual generation or gold prediction position, the query is constructed from that position's real layer input x. Each fact's three L2-normalized keys are compared with the query by cosine similarity. Its fact score is `logsumexp(slot_cosine / 0.05)`. The actual CPU VDB selects the top-1 fact and returns its complete `V ∈ R^(3×64)`. **All six arms use fixed 1/3 weights within the selected block**, replacing the preceding round's within-group softmax. The diagonal arm must therefore also be retrained; the preceding grouped checkpoint is not a matched baseline.

Let `z=A x` and `m=(v_0+v_1+v_2)/3`. Define shared trainable functions:

```
u(v) = (P_in v + c) / max(||P_in v + c||₂, 1e-6)
w(v) = P_out v

M_diagonal(V) = diag(m)
M_pooled(V)   = diag(m) + w(m) u(m)ᵀ
M_block(V)    = diag(m) + (1/3) Σ_s w(v_s) u(v_s)ᵀ

Δh = b ⊙ B [M(V) z]
```

`P_in/P_out` are both 64×64 and initialized to the identity. The 64-dimensional trainable bias c is sampled from a standard normal using an independent CPU random stream `seed+104729`, then divided by its own RMS, giving initial RMS1 without consuming the global initialization stream for shared parameters. Nonzero c is a declared architectural component and must not be removed or changed after inspecting results. b is zero-initialized. There is no additional trainable gate, manual outer multiplier, or post hoc amplitude rescaling.

Each outer arm adds `2×64²+64=8,256` parameters, totaling 2,042,624, versus 2,034,368 for diagonal. The central parameter-matched comparison is **block_outer versus pooled_outer within the same regime/seed**. Differences from diagonal may reflect both added parameters and operator structure and must be reported separately. The pooled outer term has rank≤1 and the block outer term rank≤3. The common diagonal term means total M can still have rank 64: neither complete model should be called rank 1/rank 3, and increased outer rank does not establish increased effective information capacity.

Pure FP32 K/V storage remains `3×(64+64)×4=1,536 bytes` per fact, or 24 KiB for 16 facts. No additional 64×64 matrix is stored in the VDB. P_in/P_out/c are shared parameters; reading constructs factors and applies the operator through two matrix–vector products. Block and pooled have equal parameter counts and K/V bytes, but different operator computation, so they are not FLOP-matched. CPU exact search scans all keys; sparsity concerns the selected value block, with no claim of ANN or sublinear retrieval.

Discrete top-1 fact selection is nondifferentiable, and uniform within-block weights provide no generation-CE gradient to slot scores. All arms share dense fact-group address CE, which supervises queries/keys. Addresses still come from content-dependent contextual keys; this does not fully disentangle address and content. Even when the complete block is retrieved, contributions through `u(v_s)ᵀz` can be small, negative, or mutually canceling. Uniform retrieval weights do not imply equal effective output contributions from the three slots.

## 2. New data, actual features, and training exposure

The fixed data seed is `221042`, with data protocol `block-outer-binding-data-v1`. Complete entities and payloads exclude all relevant A/B/C/D/SWAP data from reconstruction `101042` and QKV `121042`. Only historical data rules are read; new items are not selected by historical model scores. Vocabulary and templates are reused for this controlled mechanism experiment, so neither unseen vocabulary nor formats unseen throughout the research program are claimed.

Training contains 64 entities and 128 distinct three-word payloads. Each position has 16 words, each appearing 8 times. Static binds entity e to payloads `2e/2e+1`; rebind permutes all 128 payloads each epoch before assigning A/B. Each epoch has 8 steps with 8 targets per step, covering all 64 entities once and all 128 target payloads once. Target, background, and bank-order schedules are identical across the six arms within a seed. Regime changes bindings, not total target-payload exposure. Background residence is counted separately; equal per-payload background access is not assumed.

Each step's bank contains 8 targets and 8 other entities in randomized physical order. For each target, the A bank contains that epoch's A bindings; B replaces only that sample's target record, leaving the other 15 unchanged. The eight targets' A/B examples are batched as 16 sequences. They must not share a B bank that changes eight records simultaneously, and retrieval must not cross samples. Correct fact indices supervise address loss only and are not query inputs.

All `64×128=8,192` actual entity+payload texts must be re-encoded with the frozen model; contextual hidden states from another entity cannot simply be exchanged. The exact writer wrapper is `Remember this information: ` followed by the support text. With memory disabled, features are extracted from the three located word spans. This writer assumes an annotated/locatable three-word span in the support text; **it does not automatically discover fact boundaries in unrestricted long documents**. Projected training keys/values are recomputed as parameters change. Shared statistics use only the 128 canonical static A/B support records and 64 queries, excluding the full Cartesian product, C/D, dev/confirm, and held-out templates.

Known/dev/confirm each contain 16 records, and all training-example and evaluation banks contain 16 facts. Known uses the first 16 training entities; dev and confirm use distinct new entities. The three packets share A/B/C/D payloads row by row to isolate entity changes. A/B are complete payloads present in training, but under rebind a particular fixed entity–payload pair may never have received target supervision. Actual binding exposure must be counted; not all known A/B cases may be called “supervised seen bindings.”

C/D are distinct new complete payload combinations absent from training. Row i replaces only word `i%3` in A, retaining the other two, giving position counts 6/5/5. All component words have been trained. The first letter of CC/HC/CH/HH denotes support and the second query. Held-out status is relative to this training round; these templates have historical use. D/confirm features may be encoded in advance with frozen weights, but their generation/scoring must not affect gradients, statistics, early stopping, or model selection.

## 3. Fixed training and costs

Each model receives **2,048 updates, batch 8 targets, and 16 A/B sequences per step**. Only the endpoint checkpoint enters primary evaluation; C or D must not select intermediate checkpoints. Per model this is 256 epochs, 16,384 targets, 32,768 sequences, and 2,048 batched backbone forwards. Each entity has 256 target exposures, and each payload 256 supervised sequence exposures. Across 18 models: **36,864 updates, 294,912 targets, 589,824 sequences, and 36,864 training backbone forwards**.

The only loss is `full-answer CE including EOS + 0.2×dense fact-group address CE`. Both terms first average over valid prediction positions within each sequence, then weight all 16 sequences equally, giving equal A/B weight. There is no KD, hidden-state, OPD, additional reconstruction, orthogonality, entropy, or load-balancing loss. Teacher forcing refers to gold answer prefixes, not distillation from a teacher model.

Adam learning rates are `1e-4` for Wq/Wk/slot_position, `3e-4` for Wv, B, P_in/P_out/c, and `0.005` for b; betas are `(0.9,0.999)`, epsilon `1e-8`, weight decay 0, and clip norm 1 per parameter group. Both outer arms put their added parameters in the same separate optimizer group. There is no additional warmup, resumed training, training until losses match, or tuning after results.

Record per-step training CE/address loss, parameter-group gradients, mappings and target/background exposure, tokens and padding, actual input positions, backbone calls, and time. At step 1 and every 128 steps, captured actual gold-prefix positions may support no-gradient readout diagnostics of residual/input scales, without additional LM forwards or objectives. Equal budgets do not guarantee equal optimization. If rebind remains unconverged, report that alongside structural conclusions; failure does not establish a capacity impossibility. Report actual tokens, computation, and parallel elapsed times; process seconds are not exclusive GPU latency.

## 4. Teacher qualification, actual online updates, and fixed evaluation

The teacher is the frozen original model, receiving the current target's single raw support text and the same question. It does not access memory, receive additional gold annotations, or participate in training. Before formal training, canonical 64 A/B preflight must score 128/128. Development/confirmation cover all corresponding non-SWAP conditions. Retain both full-sample denominators and teacher-qualified subsets; do not remove failed items or change prompts based on confirmation results.

All student shared parameters are frozen. Its prompt contains only the question, and it uses the actual CPU VDB. Each phase first generates 16 initial A answers. Each subsequent target/world independently starts from a clean A bank and executes `A→X→A`. Each write changes only the target's three slots; all nontarget keys/values/timestamps remain strictly unchanged. Restoration returns target K/V to A; its timestamp may increase. Each intervention generates target X, the fixed neighbor `(i+1)%16` under A, and the restored target A. The neighbor's before value uses that phase's initial A result.

Non-SWAP worlds also run value-shuffle and empty controls. Shuffle preserves all keys, using a fixed fact cycle with no fixed points to move complete three-value groups while preserving within-group order. Empty truly disables every memory path, producing an added output of exactly 0. SWAP writes donor `i^1`'s A content to the target without changing the donor; it does not exchange two records simultaneously. Decoding is fixed greedy with at most 32 new tokens; save text, tokenIDs, and budget truncation.

| Packet / stage | Phases and interventions | Generations per model | Independent interventions | Restorations | Teacher generations, shared once |
| --- | --- | ---: | ---: | ---: | ---: |
| known / development | CC: B/C/SWAP | 224 | 48 | 48 | A/B/C 48 |
| dev / development | CC: B/C | 176 | 32 | 32 | A/B/C 48 |
| known / confirmation | CC: D | 96 | 16 | 16 | A/D 32 |
| confirm / confirmation | B/D in each of CC/HC/CH/HH | 704 | 128 | 128 | A/B/D in each, totaling 192 |
| Per model |  | **1,200** | **224** | **224** | — |

Across 18 models there are **21,600 student generations, 4,032 independent interventions, and 4,032 restorations, totaling 8,064 real group writes**. Teacher generations comprise 128 preflight +96 development +224 confirmation =448, giving 22,048 formal generations overall. There are 96 formal child jobs:18 train+72 student eval+5 teacher+1 prepare; smoke runs are separate. Initial bank population and control-bank materialization do not count as online logical writes. Every planned architecture/seed endpoint is evaluated; development results do not discard failing arms. There is no post hoc seed selection or winner substitution.

## 5. Metrics and criteria for further research

Normalized full-answer EM is the primary content metric; raw text is also retained. Known C/D `update_restore` requires a correct new-content answer and a correct A answer after restoration. Also report joint correctness at all three times: initial A, new content, restored A. Report new-entity confirm D in CC and HC/CH/HH in full; do not select the best format or report only R@read. A/B pair requires correct initial A and updated B. locality_joint requires correct neighbor answers both before and after; unchanged wrong→wrong text does not pass.

The main structural effect is `block_outer − pooled_outer` within the same regime/seed; diagonal is an additional reference. C serves the prescribed development description and D final confirmation. Report every architecture and seed; do not allocate extra budget based on C and subsequently claim an equal-budget D comparison. Deterministic empty-bank paired EM for different answers is logically zero, so controls use **single-world, single-sided EM differences**.

Evaluate each regime's block_outer independently against these fixed integer criteria. **Every condition must hold separately for all three seeds** to provide evidence for further investment in this block readout:

| Condition | Fixed threshold |
| --- | --- |
| Known CC C update_restore versus same-seed pooled_outer | Increase of at least 4/16 |
| Known CC D update_restore versus same-seed pooled_outer | Increase of at least 4/16 |
| block_outer known CC A/B pair | At least 12/16 |
| block_outer known CC C single-sided real−shuffle and real−empty | Each at least 4/16 |
| block_outer known CC D single-sided real−shuffle and real−empty | Each at least 4/16 |
| Corresponding teacher qualification | All relevant teacher conditions correct; otherwise qualify the conclusion |

These criteria do not guarantee statistical significance, and 1–2 correct cases in one seed do not solve generalization. They also do not automatically authorize corpus expansion. At this round's end, first report whether the criteria were met, cross-entity/format results, and locality before deciding the next round. No numerical new-entity threshold is preregistered; do not add a favorable one after seeing scores. Poor new-entity D or locality must explicitly restrict claims of “solved generalization,” even if the table's criteria pass.

Fixed secondary metrics are word-level correctness, correctness of the edited word while preserving the other two, complete-answer containment, fractions reproducing old A/B/any training payload, word counts, budget_hit, NLL, and fact recall at the first prediction and during decoding. Separate strict three-word diagnostics from diagnostics requiring only that a word position exist. State whether NLL includes EOS and how it is reduced; do not directly compare it with EOS-inclusive training CE. Three seeds share data;16 rows, multiple phases, and multiple models are not independent corpus samples.

## 6. Audit and interpretation boundaries

Save the actual query for every generated token, selected-group indices/scores/weights, and bank hashes, aligned to the true first prediction and decode steps. Reconstruct selected values bitwise from saved encoded banks, write events, and indices rather than duplicating them in each JSON trace. A CPU audit with complete queries can replay group selection over the full bank. Selected scores alone allow weight/index checks, not proof of global top-1 selection. When all three slots are retrieved uniformly, slot mass describes retrieval weight and is not a causal account of effective content contribution. Save initial banks and restoration events; check parameter hashes before/after, control permutations, and nontarget invariance.

Comparing the two outer arms changes both **whether nonlinear normalized factors are generated before or after pooling** and the outer term's maximum rank. Results cannot be attributed solely to “preserving a matrix” or a particular rank. Both are dynamic parameter operators and ultimately produce a latent vector; vector output itself does not prove insufficient information. Moving shared B across summation does not add expressivity. Even failure here constrains only this single-layer, three-slot, linear-factor, rank 64, fixed-budget small task; it does not reject multihead or deeper readers, or attention memory in general.

Before sealing, validate historical exclusions/actual span encoding,8192 texts and train-only statistics, same-seed shared and outer initialization, per-epoch exposure, sample-isolated banks, tensor/CPU formula equivalence, the shared diagonal term, zero empty-bank output, effective outer gradients, same-token routing, teacher preflight, and actual-model smoke runs. Then record source/data/plan/document SHAs. Substantive bugs require versioned fixes with original artifacts preserved; invalid runs must not be counted as negative experimental evidence.
