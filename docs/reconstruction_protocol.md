# Small-data content reconstruction and the write/read interface: fixed protocol

> English translation of the historical protocol, not a new preregistration. Original protocol hashes and experiment registrations remain immutable. Use the pinned historical source and instructions in [reproducibility.md](reproducibility.md) to reproduce sealed runs.

**Historical status: V 1, sealed before formal training on 2026-10-06.** The teacher achieved 16/16 in each training A/B world; a batched real-model smoke test and CPU data/module/gradient/scheduling tests passed. Formal outcomes were not yet available. Document, source, and data SHAs and preflight evidence are recorded in Workspace `plans/reconstruction_registration_20261006.json`. Any subsequent change requires a new version, preserved old results, and a reason. Thresholds must not move after inspecting sealed tests.

The starting evidence is the [cold-start results](coldstart_results.md): broader gradient coverage and lower answer NLL had not produced complete multiword answers, while the raw Wiki teacher also had extraction failures. This round therefore asks a narrower question first: **with the backbone frozen, can the current interface reliably reconstruct complete observed content on a very small dataset using actual vector reads and writes?** Only then compare writer and readout, and only afterward consider 256/1024 candidates. This is not another large-corpus pretraining round, and rank, pooling, or retrieval is not assumed to be an established cause.

## Fixed minimal design

| Item | Fixed choice | Interpretation boundary |
| --- | --- | --- |
| Backbone and interface | Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`; layer 20, rank/key dimension 64; actual CPU VDB; foundation bank disabled | Each of three seeds initializes fresh random A/B and the shared writer; same-seed controls copy identical initial tensors |
| Small training set | 16 notes, three-word payloads, A/B versions; 8 targets per step; all 16 facts in the bank | Use new note IDs, payloads, and files, not previously inspected confirmation cases |
| Main budget | 3 optimization seeds `71042/71043/71044`; 1,024 updates per model, batch 8; step 512 only for training-loss diagnostics | No C/D scoring to select a checkpoint; the sole primary endpoint is 1,024 |
| Main loss | Full-answer CE including EOS, equally weighted per sequence; plus 0.2-weighted fact-group address CE at each gold prediction position | No KD, hidden, OPD, consistency, or other regularizer; teacher used only for qualification |
| New online content | Untrained combinations C (development) and D (final confirmation) for the same 16 entities; plus 32 new dev entities and 64 new confirm entities | Component words seen, complete payload unseen; all statistics and fitting use training A/B only |
| Interventions | real / fixed deranged value permutation / empty; independent target-wise A→B/C/D/SWAP→A | Freeze all shared parameters; change only the target; start each intervention from an independent clean A bank |
| Architecture matrix | slots 1/3 × fixed/trainable B: 4 arms × 3 seeds = 12 models; top-4 fixed at slot level | Run the 3 baseline seeds first, then the other 9 models; no top-12 or larger-data diagnostic |

All arms use Adam: Wq/Wk LR `1e-4`, Wv `3e-4`, b `0.005`, trainable B `3e-4`; clip each parameter group's gradient norm separately to 1. Adam betas are `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0`. Data seed is `101042`. Templates, control permutations, and actual cost fields are checked against implementation and recorded in sealed manifests, not chosen from confirmation outcomes.

## Data: reconstruct content rather than infer labels from questions

A canonical support can look like `Stored note <ID>. Content: <word1> <word2> <word3>.`; the question requests the specified note's complete three-word content. Questions must not contain answer words, a left anchor, an A/B/C world marker, or a version cue. Each fact's complete A/B/C contents differ and do not duplicate other records. IDs and answers are independently randomly paired; serial number, ordering, word length, and input position must not encode an answer shortcut.

This is supervised memory writing and reconstruction: the complete observed support legitimately contains the target payload. The student's inference prompt contains only the question, without support, answer annotations, or retrieved text; the writer may read the entire observed note. Memory increments are disabled during feature extraction, and the backbone stays frozen. S1 keys and values both use the mean over the payload-content span; S3 keys and values both use the corresponding word-span means. Spans are determined by observed text, not by the question or prediction. Address labels are offline supervision only and cannot be passed directly to queries or VDB search. S1 does not retain the previous round's last-token key, so cross-round differences cannot be interpreted as a data-size-only comparison.

To test composition rather than memorization of full labels, the three word positions use explicit vocabularies. Each position's words have sufficient coverage in training A/B; the complete A/B sequences belong to the training set. C/D are fixed samples from unseen complete combinations, using the same training vocabularies so that unseen-token/tokenizer learning is not mixed into the first stage. Report per-position word frequencies, full-combination frequencies, token lengths, and first-token collisions in advance. C/D must not enter encoder centering, writer fitting, address training, optimizer updates, or checkpoint selection.

Training traverses the 16 targets in fixed shuffled epochs. Each step batches 16 sequences in fixed A-before-B order; both worlds count toward balanced target exposure. Every sample has its own bank. Its counterfactual bank is constructed from the common A bank by replacing only that sample's target, with all other records still A. Do not replace all batch targets simultaneously or retrieve across samples. Bank order follows data rows; array position is not a query/writer feature, while target order is shuffled. Training steps, target schedules, and A/B order are identical across same-seed architecture arms. A/B samples and repeated training do not increase the number of independent facts.

## Splits and teacher qualification

Record these four questions separately rather than combining them into one generalization score:

1. **Seen-fact reconstruction:** the 16 training entities and training A/B contents, with canonical support/query, test whether the actual training task can be fitted.
2. **Online writing of novel combinations:** the same 16 entities; C is used only for endpoint development diagnostics and candidate selection at step 1,024, while D remains sealed until final confirmation. Canonical format stays fixed. This tests beyond a fixed entity-to-training-label mapping, but not unseen vocabulary.
3. **New entities:** 32 dev and 64 confirm IDs, disjoint from training and each other; their complete A/B payloads are absent from training. Dev tests CC only; confirm tests CC/HC/CH/HH. Report these separately from same-entity novel-combination writes so that entity recognition and content reading are not conflated.
4. **New expression:** known 16 also tests A/B under HC/CH/HH; new-entity confirmation tests all four phases. New formats are absent from training. The same fact across formats is not an independent sample. Format generalization is not required to establish learnability of seen facts and must be reported separately.

The teacher is the same frozen backbone, given only the current target's single support and the same question. Do not append gold labels or explicit answer-location annotations. Preflight prompts/data using training A/B only, requiring complete free-generation correctness of 16/16 in each world and strict paired 16/16. Use fixed decoding/token budgets and retain inputs and outputs. If preflight fails, stop training, repair prompts/data, and seal a new version; do not remove teacher failures to obtain a perfect score. Report C/D, new-entity, and new-format teacher qualification with their evaluations without using outcomes to replace questions beforehand.

C and dev may select one candidate under the fixed rule; D and confirm cannot tune parameters, stop training early, change templates, or replace that candidate. Teacher-ineligible conditions retain full denominators, alongside a teacher-paired-eligible subset. Qualification affects interpretation, not the student's actual score. The teacher contributes no loss; it checks whether free generation with support can solve the task.

## Stage one: a stable reconstruction positive control

The baseline trains Wq/Wk/Wv and shared b offline; A/B and the backbone remain fixed. Foundation slots are disabled, without ineffective foundation retrieval calls. Each of 8 targets gets separate A/B banks, yielding 16 teacher-forced student sequences. The formal implementation may batch them into one backbone forward, with retrieval always isolated per sample. Teacher forcing means supplying the gold prefix, not running teacher distillation. Full-answer CE including EOS equally weights 16 sequences; eight A and eight B samples preserve equal world weights. Earlier two-forward smoke tests remain archived but are excluded from formal costs.

At each gold prediction position, address supervision sums the full-slot-bank softmax probability assigned to the target fact and takes its negative log: one slot for S1, three for S3. The label is a fact group; it neither forces particular answer tokens to particular word slots nor provides fact IDs to inference queries. Total loss is `CE + 0.2 * fact_group_address_CE`. Like CE, the address term first averages valid gold prediction positions, including EOS, within each sequence, then equally weights sequences and the A/B branches. It does not pool all tokens from unequal-length sequences into one mean, nor triple S3's loss weight through repeated labels. There is no KD, OPD, or hidden-state alignment.

Training batch size is 8, with equal target exposure per fact within each epoch. At 512 steps each target has 256 exposures; at 1,024 it has 512. Count CE/address tokens per world separately; this is not compute-matched to a previous one-update-per-fact regime. Step 512 records training-loss diagnostics only, without C/D or dev/confirm selection. Primary conclusions use the fixed endpoint, not the best checkpoint. Across 12 models, the budget is 12,288 updates, 98,304 target exposures, and 196,608 A/B student sequences; actual input/target tokens and cost come from logs.

Record full-sequence CE/NLL, free generation, first-position R@1/R@4, target-record residence per token, shared-parameter gradients, and processed positions. Lower NLL, higher address recall, or a correct teacher-forced first word does not replace complete free-generation success.

Forced target-record reads are not included in this fixed evaluation matrix, to limit diagnostic expansion. If separately authorized later, label them as auxiliary conditions with additional address access, not the real path or an accuracy upper bound.

## Interventions, restoration, and preservation of unrelated facts

For known 16 CC, each target starts from an independent copy of the clean A bank. After the initial A read, B/C/D/SWAP each perform a single-record write and restoration to A; branches do not accumulate, and D opens only at final confirmation. At each time point, freely generate for the same question and save full text, normalized EM, token counts, EOS/truncation, actual VDB traces, and bank hashes. SWAP writes a fixed other entity's A payload into the current target, changing only the target and leaving the donor intact. It diagnoses reassignment of seen content, not novel combinations. This semantics has been checked against implementation.

Writes/restorations atomically replace every slot of the target record. All other records' keys, values, and timestamps must remain exactly unchanged. Target timestamp/version may advance during restoration, so compare restored payload/key/value and the final answer rather than requiring the timestamp-inclusive whole-bank hash to equal its original value. Shared-parameter SHA must be unchanged across online evaluation. Every inference call reads a committed bank; it must not append the answer being predicted during prediction.

Each update adds one predetermined non-target neighbor query, indexed by `(i+1)%N`, asking about the same untouched record before and after the update. Report both joint correctness and prediction preservation; identical wrong answers at both time points are not successful preservation. One neighbor per target is a sparse locality probe, not proof that all other 15 facts are unaffected. Fixed mappings and SWAP donors are recorded in the data manifest, never selected by predictions.

Value permutation uses `Random(111042+i)` to shuffle fact indices and connect them into a single cycle, yielding a fixed derangement while keys and all other inference settings remain unchanged. Three-slot arms move entire fact groups, preserving within-group order. Empty removes dynamic memory; the foundation bank is genuinely disabled, so this represents no memory. Known CC B/C/D each test 16 shuffle/empty cases; A and SWAP add no such controls. Known HC/CH/HH test only A/B without controls. New-entity dev 32 tests CC A/B, and confirm 64 tests A/B in all four phases.

Compare **single-world full-answer EM** between real memory and controls, particularly for novel C/D. An empty bank produces one deterministic output for the same question while A/B answers differ, so its strict paired EM is logically zero and is not sufficient evidence of memory dependence. Empty generation may be cached once per question and scored against B/C/D separately, but it must be recorded as one shared generation; do not invent independent calls or undercount scoring cost.

## Fixed gates, candidate selection, and stopping rules

The following gates are fixed for this round. Small-sample rates are operational progress thresholds, not population-level confidence claims; three seeds sharing this dataset do not create 48 independent facts.

| Stage | Gate required separately for all three seeds | Permitted conclusion/action if it fails |
| --- | --- | --- |
| Teacher and implementation G 0 | Training teacher A/B each 16/16, pair 16/16; no source leakage in prompts; single-record change, restoration, frozen parameters, and CPU routing audits all pass | Repair data/implementation/prompts and reseal; do not count infrastructure failure as model zero accuracy |
| Training learnability G1 | Real seen A/B pair ≥15/16: initial A and updated B both correct; separately report restored A and all-three-time-point correctness | This recipe has not achieved stable reconstruction; finish the preset small architecture comparison without expanding data |
| Development writability G2 | C write and A restoration both correct ≥15/16; C single-world real EM exceeds shuffle and empty by at least 50 percentage points each; fixed-neighbor joint correctness ≥95% | Passing G1 alone does not establish writable memory if shared parameters merely memorize A/B |
| Final expansion eligibility G3 | The development-selected candidate, in all three seeds, has D write plus A restoration correct ≥15/16 and new-entity confirm CC A/B pair ≥80%, at least 52/64 | Do not expand to 256/1024 candidates or replace the candidate using other arms' confirmation scores; report new expression separately without retrospective selection |

G2 locality uses the 16 fixed known-C neighbor pairs as its primary denominator; ≥95% means 16/16. Do not concatenate repeated worlds into a larger independent denominator. Report other worlds' locality separately. C/update+restore requires C output and restored A to be correct; report the initial A+C+restored A triple as well, without silently substituting it for the gate. D uses the same definition.

Run the three baseline seeds first, then all three seeds of the other three architectures regardless of baseline success. These are preset bounded structural diagnostics, not an assumption that the positive control already passed. A single passing seed is not stable success. Candidate eligibility requires every seed to pass G1/G2. Among eligible arms, maximize the minimum-seed C/update+restore rate, then prefer fewer trainable parameters, then fewer vector bytes per fact; break any remaining tie by the sealed order `S1_fixedB → S3_fixedB → S1_trainB → S3_trainB`. D/confirm do not select candidates. Dev 32 separately describes new-entity performance and adds no undeclared ranking criterion.

If no arm qualifies, record “no candidate” and stop expansion. All 12 models may be evaluated and published on D/confirm after candidate identity is sealed, but G3 applies only to the three seeds of the preselected architecture. Other arms cannot replace it retrospectively. If the candidate fails G3, stop expansion without changing thresholds or seeds.

## Stages two and three: a small writer × readout factorial comparison

With matched seed, data, target/world schedule, batch, updates, shared optimizer-state rules, and decoding:

| Arm | Writer | Readout | Change and interpretation |
| --- | --- | --- | --- |
| S1_fixedB | Payload-mean keys/values, one slot | Fixed random B | Baseline positive control; reuse its fixed endpoint rather than recomputing it |
| S3_fixedB | Three word-span-mean keys/values, three slots | The same fixed B as baseline | Tests a multi-vector writer; report extra storage and candidate competition separately |
| S1_trainB | The baseline's one-slot writer | B starts at the same baseline initialization and receives gradients | Isolates additional readout freedom with a fixed writer design |
| S3_trainB | The same three-slot writer as S3_fixedB | Trainable as in S1_trainB | Completes the 2×2 interaction; do not attribute a combined difference to one factor |

S3 shares Wk/Wv across the three word spans and adds no slot-specific writer parameters. Its interpretation is limited as follows:

- The three slots have different content-dependent keys, changing both addressing and content granularity. This is a combined writer change, not an isolated demonstration that mean pooling lost information.
- The same actual layer input generates a query over all slots. Answer position, gold token, and external IDs do not select slots. The three slots are not statically averaged into one record before retrieval.
- All primary experiments use **top-4 slots**. S1 has 16 bank slots and S3 has 48, so three slots may compete for top-4. Report unique-fact recall, slot recall, and probability allocation rather than slot coverage alone. This round adds neither top-12 nor a record-group-read control.

Each arm starts fresh rather than training an already trained baseline for another 1,024 steps. Same-seed A/B, Wq/Wk/Wv, and b initial tensors are identical; constructing three slots must not consume extra randomness that changes other initial values. The statistics source remains training A/B. Learned B opens one additional gradient path while rank remains 64: it changes readout directions, not the upper bound on the layer's residual dimension. Report added parameters, optimizer state, training positions/calls/time, and actual bytes per fact. With FP32 key=64 and value=64, one slot stores 512 vector bytes and three slots store 1,536, excluding IDs, timestamps, and index overhead. Qwen's current down-projection B is 2560×64, so learning it exposes 163,840 additional trainable parameters. The matrix already existed in memory; the additions are trainable freedom and optimizer state, not another online matrix.

Equal updates/exposure do not mean equal FLOPs, time, or parameter budgets. Three slots explicitly increase memory bytes; learned B increases trainable freedom. Efficiency claims require a separately predefined resource-matched comparison rather than declaring one type of capacity superior from this small matrix. This round does not add multilayer readout, hidden loss, OPD, or a foundation dictionary.

## Before expanding to 256/1024 candidates

Expansion starts only after fixed G3 passes, with its purpose specified first:

1. **Increase only online distractors:** keep the checkpoint and target facts, add legitimate unrelated observations to reach 256/1024 records, and test retrieval/locality scaling without changing training budget.
2. **Increase training targets:** use 256/1024 new training facts and report exposure per target. Holding total steps fixed does not justify attributing every degradation to scale. Seal the choice of fixed exposure versus fixed total budget first, treating them as separate questions if necessary.

Do not merge these into a “large-corpus demonstration.” Reseal new confirmation entities, combinations, and expressions. Report free generation and intervention correctness for all targets before NLL, recall, and utilization. Preserve raw text, bank versions, individual write logs, model/parameter hashes, feature-statistics provenance, source snapshots, and actual costs regardless of outcome.

## Sealing checklist

Before formal launch, verify data seeds and split manifests; A/B/C/D combination and entity separation; support/question templates and token budgets; remaining Adam settings and address reduction; training-A/B-only feature statistics; bitwise equal same-seed initialization across four arms; fixed slot permutation and neighbor/SWAP mappings; the barrier allowing C candidate selection and only D/confirm verification; integer implementations of all gates; and source/plan hashes. After sealing, do not filter cases, replace templates, extend training, or select checkpoints based on confirmation results.
