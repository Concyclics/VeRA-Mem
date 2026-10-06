# Random Rebinding and Grouped QKV Reading: Fixed Experimental Protocol

> English translation of the historical protocol, not a new preregistration. The sealed sources and hashes remain authoritative; see [reproducibility.md](reproducibility.md).

**Version V 1, sealed before formal training.** On 2026-10-06, teacher preflight scored 128/128, four actual-model smoke groups passed, and 1,085 CPU tests passed. The coordinating agent recorded source, data, and document SHAs in the registration. No formal outcomes from this round existed beforehand. After sealing, confirmation results must not change items, templates, budgets, or thresholds. See the [QKV literature mapping](qkv_literature.md) and preceding [small-data reconstruction results](reconstruction_results.md).

This round tests whether **training each entity with multiple contents and changing sparse-memory readout allows queries constructed from current layer inputs to read newly written content, rather than merely reproduce complete training labels**. The main path remains actual VDB→value→multiplicative VeRA. Additive attention-style readout is a separate diagnostic: its success cannot count as VeRA success or establish that all stronger readers work.

## 1. Fixed matrix and shared interface

| Arm | Training bindings | Reading | Increment | Eligible main-path candidate |
| --- | --- | --- | --- | --- |
| static_flat | Fixed A/B per entity | flat top-4 slots | Multiplicative VeRA | Yes |
| static_grouped | Fixed A/B per entity | top-1 fact, then its three slots | Multiplicative VeRA | Yes |
| rebind_flat | Random reassignment each epoch | flat top-4 slots | Multiplicative VeRA | Yes |
| rebind_grouped | Random reassignment each epoch | top-1 fact, then its three slots | Multiplicative VeRA | Yes |
| rebind_grouped_additive | Same as rebind_grouped | Same grouped reading | Additive vector readout | **No; diagnostic only** |

All five arms use Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, with layer 20 down-projection input 9,728 and output 2,560, rank/key dimension 64, and temperature `0.05`. The backbone and A are fixed; B is trainable. Each fact has three actual contextual word-span vectors. Wq/Wk/Wv are shared linear projections, without multiple heads, larger rank, foundation memory, or a new value MLP. Every arm also has identical trainable `slot_position[3,64]`, initialized to zero and added after the linear key projection but before final L2 normalization. It does not change values; queries do not receive correct-slot labels.

Within a seed, all initial A/B/Wq/Wk/Wv/b and slot_position tensors are bitwise identical; only regime/read_mode/readout differ. Shared statistics are fitted on 128 canonical static training A/B support records and 64 queries, excluding dev/confirm, C/D, held-out templates, and the full entity×payload Cartesian product. Statistics are identical across all five arms. Word-slot order comes from observed text; retrieval must not be specified by the gold next word, target-array index, or a manually supplied correct slot.

### Explicit reading formulas and gradients

At every actual prediction position, `q_t=encode_query(h_t)`, where `h_t` is the current real VeRA-layer input. The three normalized slot keys are `k_{i,j}`, with cosine scores `s_{t,i,j}=q_t·k_{i,j}`, `j∈{0,1,2}`.

- **Flat:** select the four largest-cosine slots across the entire 16-fact×3-slot bank, softmax their scores divided by temperature, and mix their values. Mixing across facts is allowed.
- **Grouped:** compute `g_{t,i}=logsumexp_j(s_{t,i,j}/0.05)` for each fact and select `i*=argmax_i g_{t,i}`. Apply softmax at the same temperature only over that fact's three slots. Return their mixed value, without choosing slots using gold tokens or collapsing every fact's slots into one mean value.

Grouped top-1 fact selection is nondifferentiable. Generation CE does not pass selection gradients through this discrete choice to unselected facts; it trains only the selected three slots' scores and values. All arms additionally use identical **dense fact-group address CE**, supervising every bank key/query. Distinguish this auxiliary training objective from actual sparse reading. Group scores use three-slot logsumexp, not an independently learned entity-address encoder, so this round cannot claim complete address/content separation.

Let the mixed memory value be `m_t`. The multiplicative main path uses `Δh_t=b⊙B[(A h_t)⊙m_t]`; the additive diagnostic uses `Δh_t=b⊙B m_t`. The additive arm retains the identically initialized A buffer without using its multiplicative path; all other trainable tensors and supervision match. At step 1 and every 128 steps, replay reading without gradients at captured actual gold-prefix prediction positions, including EOS, before the parameter update. Record input, mixed-value, and residual RMS for both paths without extra backbone forwards or training losses. These training-position diagnostics do not replace free-generation reading metrics, and differences cannot be attributed to one mechanism while ignoring scale/optimization differences.

All five arms have 2,034,368 trainable parameters, including 192 slot-position parameters. Pure FP32 K/V storage is `3×(64+64)×4=1,536 bytes` per fact, or 24 KiB for a 16-record bank. Flat mixes 4 values, grouped 3: equal storage and parameters do not make their FLOPs equal. CPU exact search scans all keys; sparsity concerns value selection/mixing, without claims of ANN or sublinear search complexity.

## 2. Data, random rebinding, and the feature contract

The data protocol is `qkv-binding-episodes-v1`, with new seed `121042`. Training contains 64 entities and 128 distinct three-word payloads. Entities and complete payloads exclude all train/known/dev/confirm A/B/C/D from the preceding reconstruction seed `101042`. Vocabulary and historical templates may be reused; neither entirely unseen vocabulary nor research templates are claimed. Each of 16 words per position appears 8 times among the 128 training payloads, preventing positional word frequency from identifying a novel combination.

Static fixes entity `e` to payloads `2e` and `2e+1`. Rebind deterministically shuffles all 128 payloads each epoch, assigning the permutation's `2e/2e+1` as that entity's A/B. An epoch contains 8 steps. Its target order is a random permutation of 64 entities, taken 8 per step; thus each entity is a target exactly once and each of 128 payloads appears once as an A/B supervised target per epoch.

Each step's bank contains its 8 target entities and 8 backgrounds sampled from the other 56 entities, followed by a random permutation of all 16 physical records. Across five arms within a seed, targets, background entities, and bank order are identical, using random streams independent of regime. Changing mappings changes background content: **equal target-payload exposure does not guarantee equal background residence for each payload**. Record both, and do not interpret whole-bank content access counts as independent supervised examples.

For each target, construct separate A and B banks. Every A record uses the current epoch's A binding; B replaces only that sample's own target with its current B, preserving the other 15 A records. Do not replace all eight batch targets simultaneously or retrieve across samples. The eight A and eight B examples are concatenated in fixed order into 16 sequences for one backbone forward, each retaining an independent bank. Target fact indices are used only for training address loss, not queries or CPU search.

**The training support cache must newly encode actual texts for all 64×128=8,192 entity+payload combinations with the frozen model.** Feature extraction retains the exact writer wrapper `Remember this information: ` followed by actual support text, with memory disabled. Pool current-layer inputs separately over three tokenizer-aligned word spans. Do not directly transfer an old entity's contextual hidden states to a new entity and call it re-encoding. During training, Wk/Wv and slot_position encode required banks using current parameters, avoiding stale projected caches. Raw frozen features may be cached within the declared text contract.

### Fixed evaluation splits

All three evaluation banks contain 16 records; every training sample's bank also contains 16 records:

| Packet | Entities | A/B content | C/D content | Purpose |
| --- | --- | --- | --- | --- |
| known 16 | First 16 training entities | Their canonical static training payloads | New complete combinations | Examine seen bindings/new-content writes |
| dev 16 | 16 new entities | Same corresponding rows as known | Same corresponding rows as known | Development entity transfer/candidate selection |
| confirm 16 | Another 16 new entities | Same corresponding rows as known | Same corresponding rows as known | Final entity-transfer validation |

Known/dev/confirm deliberately share content row by row; evaluation answers are not disjoint across these splits. New-entity A/B uses trained complete content to isolate entity changes at equal bank size and payload. Complete C/D combinations occur neither in this round's training nor in the preceding dataset, and C and D differ. Row `i` changes only word `i%3` of A, retaining the other two; edited-position counts across 16 rows are 6/5/5. Both C and D combine seen words into new complete payloads, not unseen-vocabulary generalization. D follows the same controlled one-word substitution rule, rather than an independent task distribution.

Canonical support/query templates match across splits. Held-out support/query templates are unseen only in **this round's training**; they have historical use in this research program and are not newly sealed expressions for the whole study. The first letter of CC/HC/CH/HH denotes support, the second query. All frozen features, including D/confirm, may be computed before scoring, but D and confirm generation/scoring occur only after candidate sealing and cannot affect gradients, statistical fitting, or selection.

## 3. Training budget and supervision

Optimization seeds are `81042 / 81043 / 81044`. Five arms with three seeds each give **15 models**, independently trained from their corresponding shared initialization. One arm must not receive additional training and then be compared with fewer-step controls.

Each model receives 2,048 updates with batch 8 targets, totaling 16 A/B sequences; the endpoint is the sole primary checkpoint. Per model:256 epochs,16,384 target exposures,32,768 student sequences, and 2,048 batched backbone forwards. Each entity has 256 target exposures, and each training payload 256 supervised sequence exposures. Across 15 models:30,720 updates,245,760 target exposures,491,520 sequences, and 30,720 training backbone forwards. Record actual tokens, padding, input positions, and time; equal sequence counts do not imply equal tokens/FLOPs.

The only loss is `full-answer CE including EOS + 0.2×fact_group_address_CE`. Both terms first average over valid gold prediction positions, including EOS, within each sequence, then weight all 16 sequences equally. Eight A and eight B examples give equal world weights. The address objective uses the summed probability of the target's three slots without assigning each answer token to a particular word slot. There is no KD/FKL, hidden-state, OPD, style/AP, replay, load-balancing, or other new loss. Teacher forcing means gold prefixes, not teacher-model distillation.

Adam uses `1e-4` for Wq/Wk/slot_position, `3e-4` for Wv, `3e-4` for B, and `0.005` for b; betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`, and clip norm 1 per group. slot_position shares the Wq/Wk group; other groups follow the common rule. Zero-initialized b makes every path's initial increment zero; this does not permit an undeclared warm start. Intermediate training diagnostics are recorded, but C/D cannot select early stopping or a best checkpoint.

The preceding round had 512 target exposures per training entity; this round has 256 and changes training entity/content counts, slot-position parameters, and schedule. Cross-round differences therefore are not causal effects of QKV alone. Main effects are compared only within this round's matched matrix. If seen reconstruction remains unstable, complete the scheduled small matrix and report it without adding budget to reach the threshold.

## 4. Teacher qualification and online evaluation

The teacher is the frozen original model, receiving the target's single raw support text and the same question, without VDB access, extra gold labels, or answer-location annotations. It contributes to no training loss. First run preflight A/B for all 64 canonical static training entities, requiring 64/64 for each world and 64/64 paired. Failure stops formal training; fixes require a separately sealed version, not deletion of teacher-failed items.

Development/confirmation teachers cover corresponding non-SWAP worlds and phases, preserving source text, predictions, token statistics, and evidence of no memory access. For unqualified teacher conditions, retain full student denominators and report teacher-paired eligible subsets separately, without selecting items to manufacture high scores. An insufficient confirmation teacher cannot automatically establish VeRA generalization failure or justify changing the prompt afterward.

All student shared parameters are frozen. Only writer outputs from actually observed support are written to the real CPU VDB; the student prompt always contains only the question. Each phase first generates 16 A answers. Every target/world then starts from an independent clean A bank and executes A→X→A. All other keys/values/timestamps remain bitwise unchanged; the target's three slots are replaced atomically. Restoration returns target K/V to A, while timestamps may increase.

Each intervention generates the new target X, the fixed neighbor `(i+1)%16` under A, and the restored target A. The neighbor's before result uses baseline A for the same ID/phase. Except for SWAP, also generate target X under value-shuffle and empty controls. SWAP writes entity `i^1`'s A content to the target, leaving the donor unchanged. It is neither a true two-record exchange nor an unseen combination. Every control preserves its declared keys/templates/decoding parameters.

Shuffle moves entire three-slot facts while preserving within-group order and all keys, using a fixed cycle without fixed points. Record its random seed and permutation in data/evaluation manifests, without prediction-based selection. Empty removes all memory, including foundation memory. Log every call. Any future cross-world caching must use an explicitly versioned cost definition rather than fabricate multiple independent generations; this round's budget counts each generation in the table.

### Fixed evaluation matrix and call counts

Every generation allows at most 32 new tokens with fixed decoding settings, identical for student and teacher.

| Packet / stage | Phases and interventions | Student generations per model | Independent interventions | Restoration writes | Teacher generations, shared once across models |
| --- | --- | ---: | ---: | ---: | ---: |
| known / development | CC: B, C, SWAP | 224 | 48 | 48 | A/B/C:48 |
| dev / development | CC: B, C | 176 | 32 | 32 | A/B/C:48 |
| known / confirmation | CC: D | 96 | 16 | 16 | A/D:32 |
| confirm / confirmation | B, D in each CC/HC/CH/HH | 704 | 128 | 128 | A/B/D in four phases:192 |
| Per-model student total |  | **1,200** | **224** | **224** | — |

Each phase has 16 baseline A generations. Each non-SWAP world has `16×(updated+neighbor+restored+shuffle+empty)=80`; SWAP has 48. Across 15 models: **18,000 student generations,3,360 independent interventions,3,360 restorations, and 6,720 real group writes**. Teacher preflight 128 + development 96 + confirmation 224 gives **448 teacher generations**, or 18,448 formal generations overall. Initial bank population and control-bank materialization are not online logical writes; their costs are separate, as are smoke runs.

## 5. Primary metrics, candidate sealing, and stopping rules

The primary metric is normalized full-answer EM, using the existing NFKC/case/punctuation/whitespace normalization and retaining raw text. Seen A/B pair requires correct initial A and updated B. C/D `update_restore` requires correctness after writing new content and after restoring A. Joint correctness at all three points—initial A, new content, restored A—is reported separately and does not replace the threshold. `locality_joint` requires correct neighbor answers both before and after; identical but wrong answers fail.

For the same question in different worlds, a deterministic empty-bank answer cannot match two distinct payloads, making strict empty paired EM necessarily zero. This logical zero is not evidence of memory success. Use same-world single-sided real−shuffle and real−empty EM differences.

### Development thresholds: each seed of the four main VeRA arms must pass

| Check | Fixed integer threshold |
| --- | --- |
| Known CC seen A/B pair | ≥12/16 |
| Known CC C update+restore | ≥12/16 |
| Known CC C single-sided real−shuffle, real−empty | Each ≥0.50 |
| Known CC C joint neighbor correctness | ≥15/16 (corresponding to ≥90%) |
| New-entity dev CC C single-sided full EM | ≥8/16 |

These are engineering criteria for deciding whether to expand experiments, not guarantees of statistical significance or deployment quality. All five arms finish the same budget; an early baseline failure does not cancel planned controls.

Among VeRA arms passing every threshold, choose one architecture by maximizing the minimum known C update+restore across seeds, then the minimum new-entity dev C single-sided EM, then preferring fewer trainable parameters and fewer vector bytes per fact. Remaining ties use `static_flat → static_grouped → rebind_flat → rebind_grouped`. Use only development endpoints and do not choose seeds. Additive never enters ranking. If no arm qualifies, record “no candidate.”

After all 15 training and 30 development student evaluations complete and teacher qualification is saved, seal the candidate and evidence SHAs. No confirmation child may start earlier. D/confirm results for all models may be reported, but a different model with better confirmation scores cannot be substituted afterward.

### Final expansion thresholds: only the development-selected architecture, every seed

- Known CC D update+restore ≥12/16.
- Known CC D single-sided real−shuffle and real−empty each ≥0.50.
- Known CC D joint neighbor correctness ≥15/16.
- New-entity confirm CC D single-sided full EM ≥8/16.

Report new-entity D update+restore, controls, locality, and four-phase A/B pairs in full, without adding or removing primary thresholds. HC/CH/HH describe expression changes separately; confirmation formats do not select models. No candidate, any selected seed failing, or an unqualified corresponding teacher prevents automatic expansion to 256/1024 records. If teachers are unqualified, mark conclusions limited rather than using student zeros to infer a capability ceiling. Additive passing cannot trigger main-path expansion.

## 6. Routing, content, and cost diagnostics

For each generated token, save its token ID, relation to raw text, actual query routing, selected fact, total weights of the three word slots, scores, and bank hash. Distinguish the first prediction at the final prefill position from subsequent decode steps; traces at the beginning of the question are not the first answer token. Use tokenizer decoding and cumulative byte/word boundaries to align slot access with **actually generated** word positions. Preserve uncertainty when generated and gold segmentation differ; do not force gold-based correct-slot assignments.

Grouped indices are returned in slot 0/1/2 order, not score order. Thus R@1 must use the actual highest-score/weight slot or an explicitly defined selected fact, not the list's first entry. Main routing reports include whether the target fact was actually read, grouped selected-fact accuracy, slot masses, hits when relevant words were predicted, and decode residence. Flat reads four slots and grouped three; do not describe them as having equal R@4 access.

Full-answer containment, word counts, budget_hit, edited-word positional correctness, and output rates for old A/B/any training payload are prescribed diagnostics, not sample-selection criteria. NLL is a teacher-forced auxiliary metric; specify EOS inclusion and reduction instead of conflating it with EOS-inclusive training CE. First-token or fact hits do not establish content success.

Training records target exposure and background residence for every entity/payload, per-step mappings/targets/backgrounds/bank order and digests, initial/final parameters, actual gradients, each group's LR/clip, tokens/input positions, backbone calls, and elapsed time. Evaluation saves initial banks, per-target writes/restorations, control permutations, parameter hashes before/after, K/V snapshots, and CPU routes. Summed multiprocess times are not exclusive GPU latency. Separate actual value-reading cost from full-key scans.

## 7. Interpretation boundaries and pre-sealing review

The main matrix can distinguish fixed binding versus rebind, flat versus grouped, and their interaction under this budget. A grouped effect does not establish fully separated address/content encoding. This remains a single-head, three-slot, single-layer rank 64 interface; no improvement would not reject all QKV or attention memory. The additive diagnostic changes injection form. It neither belongs to the main VeRA conclusion nor establishes multiplication as the sole cause of failure.

The three seeds share the same tiny data, vocabulary, and templates, varying initialization and schedule rather than constituting independent corpora. C/D are row-wise one-word substitutions, with content deliberately paired across known/new entities; they are not arbitrary composition, semantics, or cross-domain continual learning. Even after retrieving all three slots, within-group attention may assign tiny weight to the new word. Conversely, a hit on its slot does not ensure effective reading.

Before sealing, verify all historical exclusions and paired data; the writer prefix/span/memory-disabled contract for 8,192 actual training texts; shared train-only statistics; identical five-arm initialization per seed; exact per-epoch target entity/payload balance; sample-isolated A/B changes; CPU/tensor reading and empty-bank equivalence; group-selection gradients and dense-address supervision; actual-model preflight with 128 teacher cases and batched smoke; generation-token/route alignment; the 45-condition training/development selection barrier; and document/source/plan/data SHAs. Substantive fixes require a new version and preserved original artifacts; implementation invalidity is not model failure.
