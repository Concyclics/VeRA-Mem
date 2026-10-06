# Counterfactual memory distillation: strictly audited results

The v2 cache preflight passed: the fixed writer prefix was verified. Recomputing 64 feature groups (16 historical training facts x 4 support formats) gave minimum cosine=1.00000083 and maximum relative RMSE=0.00000000; fixed thresholds are cosine >=0.999 and relative RMSE <=0.05. Training/confirmation cache SHAs match the run records. Preparation used 0 teacher generations. Invalid v1 results are excluded; fresh confirmation entity seed is 37042.

A/B share the question but differ in the target fact; P changes only the target observation wording. The primary metric requires correct answers in both worlds; an output change alone is not success.

The common warm-up uses 256 updates; each of four arms continues for 512 updates with matching target/episode schedule SHA. Initial is the original anchored checkpoint before warm-up.

## Real retrieval across four conditions

Cells: A/B paired accuracy / A-and-P joint accuracy. Each column contains 64 facts.

| Method | canonical/canonical | canonical/heldout query | heldout support/canonical | heldout support/heldout query |
|---|---|---|---|---|
| initial | 76.6% / 76.6% | 0.0% / 0.0% | 1.6% / 1.6% | 0.0% / 0.0% |
| base | 89.1% / 85.9% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| behavior | 93.8% / 87.5% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| hidden | 90.6% / 85.9% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| mixed | 92.2% / 82.8% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |

## Retrieval, incorrect switches, and shuffled controls

R@1 is reported for both A/B worlds; swapped means answering B in world A and A in world B. Wrong/wrong includes identical incorrect outputs; changed-and-both-wrong additionally requires different normalize_answer outputs. Delta is real minus shuffled paired-switch accuracy, in percentage points with 95% fact-paired bootstrap intervals.

| Method | Condition | A/B R@1 | swapped | wrong/wrong | Changed and both wrong | Changed without joint correctness | Δreal−shuffled (pp, CI) |
|---|---|---|---|---|---|---|---|
| initial | canonical/canonical | 100.0%/100.0% | 0.0% | 3.1% | 3.1% | 23.4% | +76.6 [+65.6, +87.5] |
| initial | canonical/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| initial | heldout support/canonical | 39.1%/48.4% | 0.0% | 93.8% | 10.9% | 14.1% | +1.6 [+0.0, +4.7] |
| initial | heldout support/heldout query | 1.6%/0.0% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| base | canonical/canonical | 100.0%/100.0% | 0.0% | 1.6% | 1.6% | 10.9% | +89.1 [+81.2, +95.4] |
| base | canonical/heldout query | 0.0%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| base | heldout support/canonical | 35.9%/37.5% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| base | heldout support/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| behavior | canonical/canonical | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 6.2% | +93.8 [+87.5, +98.4] |
| behavior | canonical/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| behavior | heldout support/canonical | 46.9%/48.4% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| behavior | heldout support/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| hidden | canonical/canonical | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 9.4% | +90.6 [+82.8, +96.9] |
| hidden | canonical/heldout query | 0.0%/1.6% | 0.0% | 100.0% | 3.1% | 3.1% | +0.0 [+0.0, +0.0] |
| hidden | heldout support/canonical | 40.6%/35.9% | 0.0% | 98.4% | 17.2% | 17.2% | +0.0 [+0.0, +0.0] |
| hidden | heldout support/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| mixed | canonical/canonical | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 7.8% | +92.2 [+84.4, +98.4] |
| mixed | canonical/heldout query | 0.0%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| mixed | heldout support/canonical | 43.8%/43.8% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| mixed | heldout support/heldout query | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |

## Teacher qualification

All facts remain in the main-table denominators. Conditional scores are diagnostic; teacher correctness comes from free generation and differs from margin/hidden gates in the training loss.

| Condition | Teacher A/B joint | Teacher A/P joint | Teacher A/B/P all-correct count |
|---|---|---|---|
| canonical/canonical | 100.0% | 100.0% | 64/64 |
| canonical/heldout query | 0.0% | 0.0% | 0/64 |
| heldout support/canonical | 100.0% | 100.0% | 64/64 |
| heldout support/heldout query | 1.6% | 0.0% | 0/64 |

| Method | Condition | All-fact A/B joint | Teacher A/B-correct subset | Subset denominator | A/P joint (all/teacher-qualified) |
|---|---|---|---|---|---|
| initial | canonical/canonical | 76.6% | 76.6% | 64 | 76.6%/76.6% |
| initial | canonical/heldout query | 0.0% | — | 0 | 0.0%/— |
| initial | heldout support/canonical | 1.6% | 1.6% | 64 | 1.6%/1.6% |
| initial | heldout support/heldout query | 0.0% | 0.0% | 1 | 0.0%/— |
| base | canonical/canonical | 89.1% | 89.1% | 64 | 85.9%/85.9% |
| base | canonical/heldout query | 0.0% | — | 0 | 0.0%/— |
| base | heldout support/canonical | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| base | heldout support/heldout query | 0.0% | 0.0% | 1 | 0.0%/— |
| behavior | canonical/canonical | 93.8% | 93.8% | 64 | 87.5%/87.5% |
| behavior | canonical/heldout query | 0.0% | — | 0 | 0.0%/— |
| behavior | heldout support/canonical | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| behavior | heldout support/heldout query | 0.0% | 0.0% | 1 | 0.0%/— |
| hidden | canonical/canonical | 90.6% | 90.6% | 64 | 85.9%/85.9% |
| hidden | canonical/heldout query | 0.0% | — | 0 | 0.0%/— |
| hidden | heldout support/canonical | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| hidden | heldout support/heldout query | 0.0% | 0.0% | 1 | 0.0%/— |
| mixed | canonical/canonical | 92.2% | 92.2% | 64 | 82.8%/82.8% |
| mixed | canonical/heldout query | 0.0% | — | 0 | 0.0%/— |
| mixed | heldout support/canonical | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| mixed | heldout support/heldout query | 0.0% | 0.0% | 1 | 0.0%/— |

## Paired comparisons across conditions

The four conditions for each entity are resampled together; 256 rows are not 256 independent facts. Values below are percentage-point differences from base with 95% intervals.

| Method | A/B joint | A/P joint | swapped | wrong/wrong | Unrelated-entity joint (two conditions) |
|---|---|---|---|---|---|
| base | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| behavior | +1.2 [-0.4, +2.7] | +0.4 [-1.2, +2.0] | +0.0 [+0.0, +0.0] | -0.4 [-1.2, +0.0] | +1.6 [+0.0, +3.9] |
| hidden | +0.4 [-1.2, +2.0] | +0.0 [-2.0, +2.0] | +0.0 [+0.0, +0.0] | +0.4 [-0.8, +1.6] | +0.0 [-3.1, +3.1] |
| mixed | +0.8 [-0.8, +2.7] | -0.8 [-2.3, +0.8] | +0.0 [+0.0, +0.0] | -0.4 [-1.2, +0.0] | +0.8 [-1.6, +3.1] |

## Actual training cost and teacher training coverage

| Stage | Updates | Pair exposures | sampled tokens | Total input tokens | Training seconds | Teacher first-token A/B joint | Valid behavior pairs | Valid hidden pairs |
|---|---|---|---|---|---|---|---|---|
| Common warm-up | 256 | 2048 | 0 | 946208 | 81.3 | 94.8% | 100.0% | 100.0% |
| base | 512 | 4096 | 0 | 1892732 | 156.9 | 94.9% | 100.0% | 100.0% |
| behavior | 512 | 4096 | 0 | 1892732 | 153.0 | 94.9% | 100.0% | 100.0% |
| hidden | 512 | 4096 | 0 | 1892732 | 158.2 | 94.9% | 100.0% | 100.0% |
| mixed | 512 | 4096 | 2676 | 2046139 | 222.6 | 94.9% | 100.0% | 100.0% |

The common warm-up is counted once in total physical cost. Each continuation arm inherits that warm-up in its full training lineage. Historical training of the initial checkpoint and feature-cache preparation are outside this cost scope.

## Scope and limitations

- 64 fresh entities, a finite 16-word answer vocabulary, two reserved structure families and one training seed; broader semantic generalization is untested.
- Only v2 caches with the historical writer prefix and a passing 64-feature training-only reproduction check are admitted. Invalid v1 results are excluded; confirmation uses fresh seed37042.
- A/B must both be correct for paired_switch_em. Changed predictions alone, swapped wrong answers, and wrong/wrong changes are not successes.
- Teacher eligibility conditions are reported alongside all-fact scores; no difficult teacher-ineligible facts are silently discarded.
- A/B/P and all four phases share an entity. Bootstrap samples target-fact clusters, not individual world/phase rows; intervals exclude seed/family uncertainty.
- Unrelated reads query the next entity under one target intervention; this limited preservation diagnostic is not a general forgetting benchmark.
- The actual writer may change the target key and value together. These are single-record observation interventions, not fixed-key value-only interventions.
- Oracle forces a value and changes the readout distribution; it is a diagnostic rather than a mathematical upper bound.
- Equal continuation updates/target schedules do not imply equal compute; mixed rollouts add work. Token totals are unpadded positions, not FLOPs.
- Physical training cost counts the common warm-up once and excludes training the inherited anchored initialization and cache preparation.
- This audit verifies prediction/write records, reconstructs CPU bank and shuffle hashes, and checks source/checkpoint/cache lineage; it does not replay model inference.

summary.json contains all real/shuffled/oracle/empty, teacher, paraphrase, and unrelated metrics, paired intervals, logged costs, and SHAs. Raw questions, observations, token IDs, entity IDs, and local paths are excluded from this public export.
