# Four-step interface experiments: results and student handoff

**All four steps are complete, but they did not establish a usable vector-memory method for new formats.** Writer changes remain at 0/128 on new-observation confirmation. After expanding to four relations and three-word answers, key consistency across three seeds and all five distillation arms score 0/256 in all four expression conditions. The teacher given the target's original text answers every example correctly. The evidence supports pausing further loss additions to this fixed single-layer random readout and first testing the expressive capacity and compatibility of its writing/reading interfaces. It does not reject all parameter-memory approaches.

This round fixes Qwen3-4B-Instruct-2507 (revision `cdbee75f17c01a7cc42f958dc650907174af0554`), layer-20 `mlp.down_proj`, frozen random A/B, rank/key 64, and top-k 4. During frozen evaluation, each token's input generates a query; actual CPU VDB values contribute to the VeRA residual. New observations are written online without updating shared weights. Offline training uses temporary differentiable GPU banks. See the [fixed protocol](interface_protocol.md). EM compares complete answers after normalizing case, Unicode, punctuation, and whitespace. Containment of the complete correct answer is reported separately and does not replace the primary metric.

## 1. Separate key and value interventions

Use 64 new development facts, retaining canonical questions and all other records. Only the target record changes to JSON or Markdown. The primary metric requires correct complete answers under both A/B fact states of the same question.

| Target record | JSON, correct pairs/64 | Markdown, correct pairs/64 |
|---|---:|---:|
| Original key + original value | 53 | 53 |
| New key + original value | 2 | 1 |
| Original key + new value | 0 | 1 |
| New key + new value | 0 | 0 |
| Teacher given original text | 64 | 64 |

Changing either keys or values causes a large decline. Original-key/new-value cases still retrieve the correct record at every first prediction position. All 512 same-key checks of first-position top-k indices and weights are bitwise identical. Failures therefore cannot be attributed only to question-style-induced addressing errors; value representations and readout compatibility also present problems. This intervention does not independently establish that the writer and reader are each faulty. Free-generation prefixes can diverge, so subsequent routes need not remain identical; total errors cannot be partitioned into two disjoint causal fractions from this intervention.

## 2. Can writer alignment repair the interface?

Compare last-linear, support-body masked-mean-linear, and last-MLP256, each with 3 seeds and 2000 updates with batch size 128. Freeze addressing, random readout, and scaling parameters. Train only the writer to align different observation expressions to the old canonical value for the same fact and A/B state. Last and mean each have 622,592 trainable parameters; MLP has 3,129,344. Mean additionally fits a fixed center buffer.

| Writer | Development CC, seed 42/43/44, each /32 | Real HC, each /32 | Canonical-key HC, each /32 |
|---|---|---|---|
| last | 22 / 22 / 22 | 0 / 0 / 0 | 0 / 0 / 0 |
| mean | 24 / 24 / 24 | 0 / 0 / 0 | 0 / 1 / 1 |
| MLP | 24 / 24 / 24 | 0 / 0 / 0 | 1 / 0 / 0 |

The first C/H letter denotes the observation and the second the question, with C/H meaning canonical/new expressions. Development queries cover 32 fixed targets in a 128-record bank. Under the predetermined ranking, real HC ties are broken by the small canonical-key HC differences, selecting mean. Selection is locked before confirmation, without switching to MLP based on confirmation scores. This selection does not imply a practical benefit for mean.

New confirmation contains 128 facts with a 32-token generation budget. Entire HC/HH banks use new expressions; CC/CH banks remain canonical:

| Model | CC | CH | HC | HH |
|---|---:|---:|---:|---:|
| Historical VeRA baseline | 112 | 0 | 0 | 0 |
| mean, seed 42 | 105 | 0 | 0 | 0 |
| mean, seed 43 | 112 | 0 | 0 | 0 |
| mean, seed 44 | 109 | 0 | 0 | 0 |
| Teacher with target original text | 128 | 128 | 128 | 128 |

All cells count paired successes out of 128. HC gains are 0 for all three seeds; CC changes are −5.47, 0, and −2.34 percentage points. The continuation gate—HC gain of at least 20 points, CC decline no greater than 5 points, and all three seeds passing—is not met. Shuffled-value and empty-memory paired EM are both 0.

Frozen addressing gives identical first-position retrieval across the four confirmation models. In HC, the historical baseline has 49/128 R@1 hits in each A/B world, yet 0 complete paired successes. Mean, seed 42 also has 49 hits on each side, with only 1/1 one-sided successes. Restoring canonical keys gives R@1 of 127/128 and 128/128 but only 1/128 complete paired successes. These keys are re-encoded from the same fact's complete canonical support, which still contains the answer; the target record is not forcibly selected. This changes addressing conditions and is neither a deployment result nor an accuracy upper bound.

Good training-set writer alignment does not transfer to new expressions. After matching the actual development targets and expressions, mean cosine to canonical targets for new-expression values is 0.127/0.058/0.335 for last/mean/MLP, versus 0.985/0.959/0.997 for canonical expressions. MLP's geometric gain does not become real generation success. Geometry diagnostics did not change selection after confirmation.

## 3. Multiple relations, multi-word answers, and key consistency

Training contains 4096 facts; confirmation contains 256 facts over 64 entities with 4 relations each. Answers have 3 words. Complete answers and entities are disjoint across splits, while component vocabulary remains shared. Training uses natural-language expressions; confirmation uses JSON/Markdown/YAML/INI. Banks contain hard negatives from other entities with the same answers and from other relations of the same entity.

Each arm initializes from the selected mean writer for its seed. Across three seeds, compare 512 updates with batch size 8 on the new task with common paired supervision versus an additional 0.05 A/B key-consistency term. The code name `decoupled` denotes only this regularizer. Key inputs still contain answers; there is no structural address/content separation. Confirmation answers are generated only after all final checkpoints are fixed.

Cells give complete paired-correct counts for seed 42/43/44, each out of 256:

| Method | CC | CH | HC | HH |
|---|---|---|---|---|
| Common paired supervision | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| + key consistency | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Shared teacher evaluated once with original text | 256 | 256 | 256 | 256 |

The harder task fails to recover new three-word answers even in the canonical format. This is not solely a format-generalization problem. Under this initialization and 512-step budget, neither common training nor key consistency establishes usable complete-answer readout. Task difficulty differs from the old 16-word task, so score differences between rounds cannot be assigned directly to one training change. The teacher qualifies in all four conditions; full and teacher-qualified denominators are both 256.

Local first-position R@1 changes are also inconsistent. In CC, averaged across A/B, key consistency moves seed 42 from 19.53% to 21.68% and seed 43 from 18.36% to 22.66%, but lowers seed 44 from 28.91% to 21.88%. None of these changes improves complete answers. The three seeds share historical initialization and data splits; they are not three fully independent architecture initializations.

## 4. Five-arm counterfactual-distillation ablation

Because the writer fails the gate, the protocol limits this stage to seed 42 with 256 additional steps per arm. All restore the same key-consistency model and Adam state at step 512, use the same sampling schedule, and evaluate fixed step 768. Common full-sequence FKL, answer CE, addressing, and paraphrase-consistency supervision remain.

| Arm | Change from common supervision |
|---|---|
| base | No additional counterfactual term |
| clip | Adds the old clip 10 behavioral difference |
| normalized | Adds a fixed-scale, unclipped behavioral difference |
| hidden | Adds hidden differences to normalized |
| on_policy | Every 4 steps, normalized uses a common prefix generated by an A/B mixture of student policies for FKL |

The scale calibrated on training facts is 71.015625; all 256 eligible teacher differences are clipped by the old clip 10 rule. The new target is `scale × SmoothL1((delta_student − delta_teacher) / scale)`. It changes both target magnitude and the Huber transition scale, so its effect cannot be attributed only to retaining teacher variation. Behavioral/hidden differences use only the first prediction position. Common CE always covers the complete gold answer and EOS. FKL normally uses gold prefixes; one quarter of on_policy updates use a common student prefix instead.

Cells are complete paired-correct counts out of 256. With one optimization seed, this is the predetermined limited diagnostic budget:

| Method | CC | CH | HC | HH |
|---|---:|---:|---:|---:|
| base | 0 | 0 | 0 | 0 |
| clip | 0 | 0 | 0 | 0 |
| normalized | 0 | 0 | 0 | 0 |
| hidden | 0 | 0 | 0 | 0 |
| on_policy | 0 | 0 | 0 | 0 |
| Shared teacher with identical inputs | 256 | 256 | 256 | 256 |

Correcting clipped targets, adding hidden differences, and using student trajectories all fail to recover complete factual answers. A/B-averaged first-position R@1 in CC is 35.94% for both base and on_policy; in HC it is 7.23% and 5.47%, respectively. Small top-4 changes in a training-routing proxy are not generation improvements.

Across all 11 models from steps three/four, four conditions, and A/B worlds, all 22,528 real-retrieval outputs have complete EM and complete-answer containment of 0. Of these, 15,299 (67.91%) already contain three words, and only 18 reach the 32-token budget. Empty and single-word output counts are both 0. Frequent errors are plausible but factually wrong phrases such as `candlelight river stone` and `forest moon sky`. Only 491 outputs (2.18%) have the correct first word. Widespread failure therefore cannot be explained solely by extra formatting or truncation, and simple regression to old single-word outputs is not observed.

Of 11,264 A/B fact pairs, 1,152 change their output while both sides remain wrong; the remaining 10,112 repeat the same wrong output. These pooled model/condition counts describe error types, not 22,528 independent samples or training uncertainty. Per-model details are in the [answer-error audit](results/interface/answer_errors.json).

On_policy uses a common student prefix of at most 8 tokens with forward KL. It neither samples independently per world nor reproduces OPCD exactly. Hidden and on_policy each add one factor relative to normalized; they do not add both simultaneously. The five arms are not compute-matched.

## Costs, verification, and limits

Each of the five arms receives 256 updates and 2048 target-pair exposures, covering 1637 distinct facts in continuation. Including the shared warm-up gives 768 updates, 6144 exposures, and 3159 distinct facts. A 4096-record training pool does not mean each arm traverses a full epoch.

| Step-four arm | Backbone calls | Unpadded input positions | Training-loop seconds |
|---|---:|---:|---:|
| base | 1,536 | 1,284,417 | 86.39 |
| clip | 1,536 | 1,284,417 | 86.35 |
| normalized | 1,536 | 1,284,417 | 86.36 |
| hidden | 1,536 | 1,284,417 | 79.90 |
| on_policy | 4,678 | 1,836,334 | 169.83 |

On_policy has 64 sampling updates, 512 trajectories, and 2758 sampled tokens, using 42.97% more input positions than base. The 11 training runs in steps three/four total 29,254 backbone calls, 22,383,360 input positions, and 1471.43 summed training-loop seconds; shared warm-up is counted once. These costs exclude writers, feature preparation, calibration, smoke tests, evaluation, and historical training of the inherited checkpoint. Calls have different batch sizes and padding is not fully recorded, so counts are not exact FLOPs. Timing in a parallel GPU environment is not exclusive serving latency.

All 22 suites and 53 jobs completed. All 62,976 generation records passed per-record arithmetic, label, provenance, and frozen-state checks. The independent CPU bank audit covers all 25 evaluations: 61,696 reads, 18,240 independent B updates, 7,346 shuffled banks, and 125 initial snapshots passed exact reconstruction and hash checks; see the [bank audit](results/interface/bank_audit.json). The frozen backbone and fixed random matrices are unchanged. Checks also confirm that the five arms restore the same Adam state, sampling schedule, and gold branch.

Step one has a separate [coverage statement](results/interface/step1_coverage.json): 64 original bank records passed bitwise round-trip checks, 128 read hashes can be reconstructed exactly, and 512 fixed-prefix route-equality checks passed. The other 896 intervention reads do not retain raw K/V patches, so only log consistency can be checked; full tensor replay cannot be claimed for them. This limitation is reported separately from the 25 later complete bank audits.

All 557 pytest tests and the new audit-script self-tests passed. Raw runs are synchronized locally with checksum verification. This round's GPU 0/1/3 processes have ended and released their memory.

The teacher receives the correct target's single original observation, while the student faces a full bank with negatives. The teacher gap therefore mixes fact localization, encoding, addressing, and readout; it is not pure compression loss or a fair text-RAG comparison. Empty memory generates one answer scored against mutually exclusive A/B labels, making paired EM necessarily 0. That alone is not causal evidence. Local changes to record bytes also do not establish unchanged behavior on other questions.

Expanded confirmation uses paired bootstrap clustered by entity and by four-entity construction group: 256 facts correspond to 64 entities and 16 groups. When all outcomes are zero, empirical resampling intervals also collapse to zero; this does not prove that the population success probability is exactly zero. Three optimization seeds share the historical backbone, projections, and data split and do not cover all training uncertainty; step four has only one seed. The training-routing proxy uses final checkpoints and frozen first-position inputs, so it cannot establish a broken gradient path throughout the sequence.

This is a limited-vocabulary mechanism experiment with exact search over small banks. It does not establish arbitrary-document memory, long-term continual learning, large-scale ANN efficiency, or exclusive inference latency. Failure under this budget does not prove that longer training or all parameter-memory architectures are ineffective.

The student handoff conclusion should be: **the current fixed single-layer VeRA interface can read and write the old single-word task in its canonical expression, but recovery of new expressions and more complex answers has not been established. Further tuning only of OPD fraction, hidden weight, or difference losses lacks supporting evidence.**

If the direction continues, first run an architectural positive control with an explicit acceptance gate: under canonical inputs, test whether a more expressive shared reader recovers complete three-word answers for new entities and clearly outperforms shuffled values. Learning shared B can serve as a capacity diagnostic, but changes standard VeRA's frozen random-readout constraint. Only after that succeeds should genuinely separate entity/relation addressing and answer-content writing be tested, followed by sparse retrieval on new expressions. These structures were not trained in this round and are not known to work. Seal the current confirmation set; future selection needs new development and confirmation splits.

## Review and artifacts

Paths below are relative to the repository root. Code and protocols are committed; backbone weights were not redundantly backed up in this round. Actual commands, source snapshots, model versions, checkpoints, and raw logs are preserved per suite.

- Local raw artifacts: `../runs/xtrah100/interface_*_20261006/`.
- Remote raw artifacts: `xtrah100:/ssd3/chenhan/VeRA-Mem-Workspace/runs/interface_*_20261006/`.
- [Full per-record summary](results/interface/summary.json), [automated report](results/interface/report.md), and [data/training/geometry/routing diagnostics](results/interface/diagnostics.json).
- `../plans/selection.json` fixes writer selection; `../plans/interface_step4_gate_20261006.json` fixes the step-four budget. Full orchestration state is in `../plans/interface_orchestration_20261006.json`.
- Final synchronization verification is in the workspace's `../backup_manifest.json`. Remote and local clocks differ; do not directly merge timestamps across hosts. Dependency order follows completed manifests and orchestration state.

After installing the project's test dependencies and obtaining local raw artifacts, review on CPU without launching remote training:

```bash
python scripts/summarize_interface.py --runs-root ../runs/xtrah100 --output /tmp/vera-interface-review
python scripts/audit_interface_banks.py --runs-root ../runs/xtrah100 --output /tmp/vera-interface-banks
python scripts/audit_interface_training_routes.py --runs-root ../runs/xtrah100 --output /tmp/vera-interface-training-routes.json
python scripts/audit_interface_answers.py --execute-final --runs-root ../runs/xtrah100 --output /tmp/vera-interface-answer-errors.json
```

The training-routing output file must not already exist. That script audits training caches only and does not read confirmation data. The answer audit requires all 11 expanded confirmation runs to finish. Bank and training-routing audits require PyTorch. Final confirmation must not become another source for hyperparameter or checkpoint selection.
