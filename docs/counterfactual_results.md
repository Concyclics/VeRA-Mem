# Counterfactual context distillation: confirmation results and student handoff

The valid experiment is complete. **Counterfactual behavioral differences, hidden differences, and mixed student trajectories did not solve generalization in the new-observation/original-question condition: the teacher with original text answers both sides correctly for 64/64 pairs, while all four trained arms score 0/64.** Paired accuracy increases in the original format, but auxiliary losses add only 1–3 pairs over the paired-training baseline. Every incremental interval from the predetermined entity bootstrap includes 0, so a stable benefit is not established.

In the other two conditions involving new questions, the teacher frequently emits JSON/tables and truncates at the 4-token budget. Their zero scores cannot be attributed solely to VeRA. All examples and teacher checks are retained below, rather than only filtered scores.

## What was tested

Use fixed Qwen3-4B-Instruct-2507, layer-20 `mlp.down_proj`, rank/key dimension 64, top-k 4, and the historical anchored initialization. Read three independent banks with exactly the same question: A contains the original fact; B replaces only the target record's answer; P changes only that record's expression while preserving the fact. The normal writer generates both key and value. Bytes and timestamps of all other records stay unchanged.

The teacher receives the corresponding observation text. The student prompt omits it and uses token-wise CPU VDB retrieval followed by VeRA inference. Shared writers/readers are trained offline. All model parameters remain frozen during confirmation, which performs only vector writes or replacements.

There are 4096 training facts, with a shared 256-update batch-8 warm-up. Each of four arms continues for 512 steps from the same module and Adam state. Final checkpoints are fixed, without selection on confirmation scores. New confirmation answers are generated only after all formal training ends. The new set has 64 entities, seed 37042, disjoint from historical sets and invalid-version seed 27042. JSON and Markdown are held-out confirmation structures, with each A/B label balanced across four query×support combinations.

| Arm | Addition to common paired supervision |
| --- | --- |
| base | A/B/P forward KL, first-token labels, actual and cross-expression addressing, A/P consistency, and canonical replay |
| behavior | Adds a 0.1 teacher–student counterfactual logit-margin difference loss at the first prediction position |
| hidden | Adds a 0.1 relative MSE on normalized hidden differences to behavior |
| mixed | Every 4 steps, samples a common prefix from an A/B mixture of student policies, replacing one gold-prefix FKL update in hidden; correct first-token labels remain |

All four arms use paired data. The comparison identifies increments from auxiliary losses and trajectory recipes, not the isolated benefit of adding paired data. Full definitions and weights are in the [fixed protocol](counterfactual_protocol.md). Code entry points are `prepare_counterfactual.py`, `vera_mem.counterfactual_run`, and `run_counterfactual_suite.py`.

## Main results

Cells count **facts answered correctly in both worlds A and B, out of 64**. A/B answers differ; output changes alone are not success. Columns use the same 64 targets, so an entity appearing in multiple columns is not an independent sample each time.

| Model | Original observation/original question | Original observation/new question | New observation/original question | New observation/new question |
| --- | ---: | ---: | ---: | ---: |
| Anchored, before training | 49 | 0 | 1 | 0 |
| Paired-training base | 57 | 0 | 0 | 0 |
| + behavioral difference | 60 | 0 | 0 | 0 |
| + hidden difference | 58 | 0 | 0 | 0 |
| + mixed student trajectories | 59 | 0 | 0 | 0 |
| Teacher with original text | 64 | 0 | 64 | 1 |

Original-format gains over base are 3/64 for behavior, 1/64 for hidden, and 2/64 for mixed. The predetermined entity-paired bootstrap jointly over four conditions (2000 resamples, seed 123) gives mean gains of +1.17, +0.39, and +0.78 percentage points, with 95% intervals [-0.39,+2.73], [-1.17,+1.95], and [-0.78,+2.73], respectively. All include 0. This is the planned statistical comparison, but averaging conditions that include teacher-format failures is not a pure semantic-ability score. The intervals also exclude training-seed and new structural-family uncertainty.

Shuffled-value and empty-memory paired accuracy is 0 for every model and condition. Real original-format readout does depend on memory. Paired update capability under new observations has not been established.

## How teacher checks constrain interpretation

| Observation/question | Teacher A correct | B correct | P correct | A/B both correct | A/B/P all correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original/original | 64 | 64 | 64 | 64 | 64 |
| Original/new | 0 | 0 | 0 | 0 | 0 |
| New/original | 64 | 64 | 64 | 64 | 64 |
| New/new | 4 | 3 | 2 | 1 | 0 |

New questions already instruct the model to return only the word, but the teacher often emits JSON field names or the start of a table. All 192 teacher A/B/P generations in original-observation/new-question reach the 4-token limit and fail strict EM. Whether a longer budget would express the correct fact was not measured; longer output would not automatically satisfy the strict single-word format.

Thus, **the cleanest negative generalization result is new observation/original question**: the backbone can read facts from these new formats, but the current vector-memory path does not reliably replace the fact under the same question. The other two conditions have measured results but also involve teacher instruction-following and truncation problems. Teacher-qualified subset denominators of 0 or 1 cannot support stable comparisons. The full summary reports both all 64 examples and teacher-qualified subsets without filtering to inflate the main score.

## Problems extend beyond retrieval

For new observations with original questions, first-prediction correct-record R@1 and generation counts are as follows. A/B are listed separately, each out of 64.

| Arm | A/B R@1 | A/B one-sided correct | A/B both correct |
| --- | --- | --- | ---: |
| base | 23 / 24 | 1 / 2 | 0 |
| behavior | 30 / 31 | 1 / 2 | 0 |
| hidden | 26 / 23 | 1 / 0 | 0 |
| mixed | 28 / 28 | 1 / 2 | 0 |

The behavioral auxiliary increases these R@1 counts without producing generation gains. In hidden's A world, 25 of 26 initial hits still yield an incorrect answer; all 23 B-world hits are incorrect. Correct-record decode residency is approximately 12.2%/13.8%. The bottleneck therefore extends beyond first-position addressing and may include writer representations, VeRA readout, and later retrieval drift. These experiments do not fully separate them.

Oracle is not an accuracy upper bound: it forces one correct value at every prompt/decode position, changing the real top-k mixture and token-wise routing. In the original format, real/oracle paired correctness is 57/16 for base, 58/29 for hidden, and 59/34 for mixed. Oracle failure alone cannot establish that the value lacks information or that the writer is solely at fault.

## Paraphrase consistency and unrelated memory

In the original-format condition, A/P joint-correct counts are initial 49, base 55, behavior 56, hidden 55, and mixed 53. A/P prediction-agreement counts are 58, 58, 58, 58, and 55. These must remain distinct: unchanged output may simply repeat the same wrong answer.

After updating one record, predictions for another fixed entity's original-format question remain unchanged for initial 63, base 64, behavior 63, hidden 64, and mixed 64 cases. Correct-before-and-after counts are 54, 59, 61, 59, and 60. With both new formats, some arms have 64 unchanged predictions but 0 correct-before-and-after cases. Only one other entity per update and the two diagonal conditions were checked; this does not establish general absence of forgetting.

Strict counts of changed outputs with both sides wrong are stored separately from `changed_but_not_both_correct_rate`, which includes one-sided correctness. All metrics, paired intervals, and controls are in the [automated audit report](results/counterfactual/report.md) and [machine-readable summary](results/counterfactual/summary.json).

## Limits of the objectives and cost comparison

Across all four arms, the teacher jointly predicts correct first tokens on 3888/4096 continuation target exposures (94.92%), while the behavioral gate is active on 4095/4096. However, **all 4096 teacher differences exceed the preset clipping threshold, so all eligible targets become +10**. The behavioral term therefore tests a capped margin-difference constraint without retaining variation in teacher effect magnitudes. Its result cannot reject all counterfactual-distillation objectives.

Mixed performs 128 student-trajectory updates, sampling 1024 trajectories and 2676 tokens. Unpadded input positions, including teacher, student, sampling, and replay, total 2,046,139 versus base's 1,892,732, an 8.1% increase. Continuation takes 222.6 seconds versus 156.9 seconds for base. The recipe simultaneously changes FKL prefixes, lengths, and address-supervision states; sampling updates also coincide with canonical-replay updates. It is neither a pure isolated OPD effect nor a matched-FLOPs comparison.

Counting the valid shared warm-up once and all four continuations gives 18,432 target-pair exposures, 8,670,543 input positions, and approximately 771.9 seconds summed across training stages. This is not the wall-clock duration after two-GPU parallelism. Costs exclude historical training of the inherited checkpoint, feature preparation, evaluation, smoke tests, and invalid v1 debugging runs.

## Corrected preprocessing error

Version v1 omitted the historical writer prefix `Remember this information: `: A/P reused historical prefixed features, while B and confirmation features were newly extracted without it. This confounded factual changes with input-format changes. All v1 scores are invalid; raw logs, checkpoints, and source snapshots are preserved, with `INVALID_PROTOCOL.json` markers for the relevant suites.

Valid version `counterfactual-context-v2` makes all writer inputs consistent, while the teacher still receives original text. Preparation, warm-up, and all four training arms restart from the original anchored checkpoint with unchanged losses and budgets. Recomputing 16 fixed historical facts × 4 expressions gives 64 feature records exactly matching the old cache, with relative RMSE 0 throughout. New regression checks reject a missing prefix. Confirmation uses seed 37042 and excludes seed 27042 seen during debugging. No v1 score is used as research evidence in this report.

All 388 tests passed. The valid five models' 12,288 predictions were individually recomputed. A/B/P bank updates and shuffle hashes can be reconstructed; training sample/view schedules, warm/final checkpoints, model freezing, source code, and actual cache SHAs passed audit. With one training seed, a 16-word answer set, 64 confirmation entities, and two new structures, this remains a mechanism experiment.

## Recommendations for the next round

1. **Validate the teacher interface first.** Calibrate structured-question output constraints and generation budgets on training entities, reporting strict-format and factual correctness separately. Then seal new entities and structural families. Do not modify prompts on this confirmation set and select high scores.
2. **Separate the writer from readout first.** Hold the original question fixed and map different observation expressions of the same fact to fixed, readable value targets. Separately measure key consistency across A/B and paraphrases. Add controls preserving actual token-wise mixtures to complement the current forced-value oracle and diagnose later-token addressing drift.
3. **Then test counterfactual-objective increments.** On an independent development set, compare a normalized objective preserving teacher-effect scale, the current clipped objective, and a no-difference baseline, using multiple training seeds. Hold paired data, replay, and exposure budgets constant before deciding whether to increase OPD. A claim about paired data itself also requires a budget-matched unpaired-data control.

The current student handoff conclusion is: **single-record vector-memory replacement works in the original format, but these counterfactual auxiliaries do not establish generalization to new observation formats. New-question tests additionally need a valid teacher interface.** Lower hidden loss, retrieval hits, or changed outputs cannot substitute for that conclusion.
