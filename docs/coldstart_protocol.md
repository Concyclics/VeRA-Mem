# Wikipedia cold start and foundation dictionary pilot

Protocol fixed before inspecting fresh confirmation outputs, 2026-10-06.

Question: does more diverse offline data, a shared learnable foundation dictionary,
or more gradient coverage reduce failure of writable VeRA memory? This is an
exploratory single-seed pilot, not Wikipedia-scale pretraining or proof of
open-domain continual learning. The 4B backbone stays frozen at the recorded
Qwen3-4B-Instruct-2507 revision; one layer-20 down projection uses rank 64.

## Data and splits

Use the pinned English Wikimedia 20231101 parquet shard recorded in the data
manifest. It contains 156,289 articles; select 32,768 train / 64 development /
128 confirmation observations after exact duplicate and answer-overlap checks.
Split before selecting article passages. No claim of exclusion from Qwen's own
pretraining, or exhaustive semantic near-duplicate removal, is made.

Each observation contains a natural passage, a random note identity and a
three-word target. The query provides a short left anchor but never the target.
Counterfactual B replaces exactly that span with a same-split donor answer.
The matching synthetic condition preserves article identities, queries, answers
and update schedule, rendering observations as concise facts instead of prose.
Natural A is Wikipedia-derived; B is an artificial update, not an encyclopedia
fact. Actual first-token collisions are repaired deterministically within each
split and recorded. The complete observed passage is an authorized memory write;
it is not a future-token causal-language-model training example.

New synthetic facts (4,096 / 64 / 128) follow the four-relation grouping protocol
with fresh identities and answers. Only canonical view 0 is used in training.
Heldout query/support formats are excluded from all offline training; dev and
confirmation articles/entities/full answers are isolated. Training P equals A
in this pilot, so AP consistency is not an independent augmentation experiment.

## Arms and training budget

All arms start from one new seed 63042, train-only feature centers, and frozen
random VeRA A/B matrices. Common initialization uses the first 8,192 Wikipedia
training observations for normalization statistics, including the no-warm control;
thus no-warm means no gradient warmup, not no access to unsupervised statistics.
Value writer uses content-masked mean features; keys remain last-token features.

1. `no_warm`: no gradient warmup, then 512 synthetic updates.
2. `matched_warm`: 1,024 compact matched-observation updates, then 512 synthetic.
3. `wiki_warm`: 1,024 natural Wikipedia updates, then 512 synthetic.
4. `fixed`: initialize 1,024 foundation prototypes by spherical k-means on all
   Wikipedia train A keys/values from the common Wiki checkpoint, freeze them,
   then 512 synthetic updates.
5. `learned_sparse`: same initialization, jointly train prototype keys/values
   and VeRA for 512 synthetic updates with hard top-k routing.
6. `dense_warm`: same, use dense routing for the first 256 updates, then sparse.
7. `straight_through`: same, sparse forward and dense backward for 512 updates.
8. `wiki_joint`: initialize prototypes from the fresh train-only Wikipedia
   features before warmup, jointly train VeRA and prototypes on 1,024 Wikipedia
   updates with straight-through routing, then 512 synthetic updates with the
   same routing. This explicitly tests corpus learning of the foundation bank,
   in addition to initialization followed by synthetic-only learning. It changes
   both warmup routing and foundation use, so is not a single-factor contrast.

Batch size is 8 targets, 72 observed facts per training bank. Targets traverse
shuffled epochs; identical paired schedules make warmup gold-token budgets equal.
Available same-answer donors and same-entity records are included as hard
negatives; the Wikipedia data does not have the synthetic data's full grouping
structure. Report unique target and bank exposures and actual input/gold tokens.
The selected 32K corpus size must not be presented as 32K target exposures when
only 8,192 targets were used. This is a bounded scaling probe.

Common loss: context-teacher forward KL + 0.5 full-answer CE + actual-address
and view-address losses (0.2 each) + 0.05 AP JS + 0.05 A/B key consistency.
No on-policy samples or hidden-state effect loss in these comparisons. Optimizer
resets at each phase for every arm. Wv learning rate 1e-4, Wq/Wk 1e-5, b 0.005,
and trainable prototype keys/values 1e-3. Gradient clipping is per parameter group.

Episodic top-4 and foundation top-4 are addressed separately from each actual
layer input. Their rank vectors combine as v_episode + 0.25*v_foundation before
one existing VeRA gate. This accesses up to eight records, not four. Foundation
parameters are offline learned; new facts remain separately writable, never
clustered or averaged with conflicting records. Dense backward incurs dense
computation. Output-level straight-through mixing, not just straight-through
weights, is required for nonselected values to receive gradients.

## Evaluation and interpretation

Freeze every shared parameter online and force sparse evaluation in all arms.
Use the real CPU vector store for both banks, with durable before/after hashes,
explicit single-record counterfactual writes, and full-answer A/B paired EM.
Run CC/CH/HC/HH on fresh Wikipedia and synthetic confirmation: first letter is
support format, second query format. Teacher, shuffled episodic values, empty
episodic bank and canonical-key diagnostics retain explicit denominators.
Empty episodic memory does not mean empty foundation memory.

Development uses 16 targets per domain; confirmation uses 64 targets from each
128-record bank, fixed by the evaluator independently of model results. The
original corpus also has 64 development records. Report this sampling clearly.
Separately, a train-memorization probe uses 8 targets from the first 128 synthetic
training targets. It duplicates the canonical view only to satisfy the evaluator's
two-view schema: all four phase labels denote the identical trained format here.
It is not heldout generalization and its 128-record bank does not guarantee the
same hard-negative composition as training. Every selected target received a
training update. This distinguishes failure to learn the observed task from a
purely heldout-format failure.
Teacher contextual A/B accuracy should be at least 90%; below this, distinguish
dataset/teacher infeasibility from student failure. A useful pilot improvement
requires at least +10 points HC and at least 20% absolute paired EM, without more
than 5 points CC loss, but all arms run and all outcomes are reported regardless.

On the predeclared straight-through checkpoint, compare full 1,024 foundation,
no foundation, random 256 and clustered 256 on the identical fixed test cases.
This tests compression as well as density; clustering need not preserve function.
Do not RMS-renormalize merged values or quietly apply multiplicity biases.
Record route coverage, gradient coverage, retained dense mass, training cost and
value variance. Coverage improvement alone is not task generalization.

All main comparisons have one optimization seed; percentages are exploratory.
The single fixed rank-64 readout has a limited output subspace that dictionary
growth alone cannot expand. Negative results cannot rule out broader architectures,
larger pretraining budgets, deeper memory layers, or different write objectives.

Artifacts live under remote /ssd3/chenhan/VeRA-Mem-Workspace/runs/coldstart_*;
local copies preserve datasets, source snapshots, logs, checkpoints and raw writes.
No public base-model weights are copied into the local backup or Git repository.

## Data integrity correction before confirmation

The full preparation audit found one training record, index 13,896 / ID
`coldwiki-6184b88599664708a482acba`, whose answer `Ship launches Ship`
overlapped its left anchor. Replacing the first substring incorrectly removed
part of the Wikipedia B anchor. The matched observation was semantically correct.
No development or confirmation record had this issue. The faulty B was a target
at warm update 940, so both affected Wiki warm branches are excluded in full via
`analysis_excluded.json` and rerun from their common starting checkpoint.

The corrected dataset preserves every identity, answer, query and A observation;
it changes only that natural B observation, using the target offset after the
unique anchor. Only its B feature is recomputed. Index 13,896 is outside the first
8,192 rows used for normalization; initialization statistics and all A features
remain identical. The unaffected matched warmup and no-warm control are retained.
The final Wiki warm suites use names `coldstart_warm_a_v2_20261006` and
`coldstart_warm_c_v2_20261006`. Original evidence and discarded compute costs remain
available, separately from the retained experimental results.

## Teacher feasibility follow-up, fixed before confirmation

The retained no-warm development run found raw Wikipedia contextual teacher paired
accuracy CC=0/16, CH=0/16, HC=1/16, HH=0/16, versus 16/16 in every synthetic phase.
Wikipedia student zeros therefore cannot establish a memory/generalization failure.
They remain descriptive results of a failed-teacher condition. This motivated a
separate train-only check, not modification of the reserved confirmation examples.

Compare three fixed teacher-context strategies on the first 16 **eligible** training
records and 64 other eligible training records chosen by fixed seed 95043:
baseline; explicit literal
extraction instructions; explicit `<requested_span>` answer annotation. Questions
are unchanged. The last condition adds answer-label localization supervision and
must never be called raw-context distillation or unsupervised Wikipedia learning.

Eligibility is determined before model loading, never from model outputs: exclude
a record if either full A/B answer already occurs as a token-bounded normalized
phrase in its question. Normalization is NFKC, case folding, punctuation-to-spaces
and whitespace collapse; articles are preserved. Partial-word substrings alone do
not exclude a record. Keep the original cache order for calibration, sample the
remaining eligible indices once, and save every exclusion with its source index,
ID, question, provenance and reason. This rule was fixed after a pre-model guard
found train record 6295 (`David Croft and` already appears in the normalized anchor
`David Croft, and written by`), before any teacher-probe prediction was generated.
It changes probe eligibility only, not the original training data or schedules.
The literal annotation helper must also validate all 32,768 training A/B source
contexts (65,536 checks), preserving labels including `&`; any unrepresentable
annotation aborts before model loading with counts and reasons in the manifest.

Before seeing this probe's results, the follow-up rule is fixed: if the instruction
strategy has at least 90% paired validation EM, use it; otherwise use the annotated
strategy only if it meets that threshold; otherwise do not train another model with
an unqualified teacher. On a pass, one supplementary `wiki_teacher_repair` arm uses
the common initialization, 1,024 Wiki updates with the selected teacher strategy,
the same 1,024-prototype initialization, and 512 synthetic straight-through updates
with the ordinary synthetic teacher. Student observations, feature caches, targets,
sample schedule, inference and optimizer settings stay identical to the main
straight-through arm. Evaluate both dev domains, both confirmation domains and
the train probe with the same cases. Report it separately as a follow-up to the
teacher failure; original eight-arm evidence and their limitations remain visible.
