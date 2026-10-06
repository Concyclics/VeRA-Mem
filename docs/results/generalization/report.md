# Template generalization experiment summary

Generated at (UTC): 2026-10-05T18:15:18.985574+00:00

Include only completed suites/runs with predictions consistent with statistics. The four quadrants independently vary query and observation formats.
Heldout-query columns equally average XML, CSV, and dialogue templates. The three views refer to the same facts and are not independent samples.

| Condition | selected / updates | canonical query + canonical support | heldout query + canonical support | canonical query + heldout support | heldout query + heldout support |
| --- | ---: | ---: | ---: | ---: | ---: |
| augment | 1536 / 1536 | 24.2% | 4.2% | 0.0% | 0.8% |
| canonical | 1536 / 1536 | 100.0% | 0.0% | 0.0% | 0.0% |
| invariant | 1536 / 1536 | 33.6% | 3.9% | 0.0% | 0.3% |

Real retrieval, forced-correct-value, shuffled-value, and zero-residual diagnostics for each quadrant:

| Condition | Quadrant | Real | Forced value | Shuffled | Empty |
| --- | --- | ---: | ---: | ---: | ---: |
| augment | canonical_query/canonical_support | 24.2% | 46.9% | 2.3% | 0.0% |
| augment | heldout_query/canonical_support | 4.2% | 6.5% | 4.2% | 0.0% |
| augment | canonical_query/heldout_support | 0.0% | 0.0% | 0.0% | 0.0% |
| augment | heldout_query/heldout_support | 0.8% | 0.0% | 0.0% | 0.0% |
| canonical | canonical_query/canonical_support | 100.0% | 63.3% | 0.0% | 0.0% |
| canonical | heldout_query/canonical_support | 0.0% | 1.0% | 0.0% | 0.0% |
| canonical | canonical_query/heldout_support | 0.0% | 0.0% | 0.0% | 0.0% |
| canonical | heldout_query/heldout_support | 0.0% | 0.3% | 0.3% | 0.0% |
| invariant | canonical_query/canonical_support | 33.6% | 53.9% | 5.5% | 0.0% |
| invariant | heldout_query/canonical_support | 3.9% | 7.3% | 3.9% | 0.0% |
| invariant | canonical_query/heldout_support | 0.0% | 0.0% | 0.0% | 0.0% |
| invariant | heldout_query/heldout_support | 0.3% | 0.5% | 0.0% | 0.0% |

Heldout query templates, using real retrieval:

| Condition | Support bank | Query template | EM | Recall@4 |
| --- | --- | --- | ---: | ---: |
| augment | canonical_support | test_query_00 | 0.0% | 3.1% |
| augment | canonical_support | test_query_01 | 6.2% | 3.1% |
| augment | canonical_support | test_query_02 | 6.2% | 6.2% |
| augment | heldout_support | test_query_00 | 0.0% | 5.5% |
| augment | heldout_support | test_query_01 | 2.3% | 1.6% |
| augment | heldout_support | test_query_02 | 0.0% | 5.5% |
| canonical | canonical_support | test_query_00 | 0.0% | 3.9% |
| canonical | canonical_support | test_query_01 | 0.0% | 3.1% |
| canonical | canonical_support | test_query_02 | 0.0% | 3.1% |
| canonical | heldout_support | test_query_00 | 0.0% | 2.3% |
| canonical | heldout_support | test_query_01 | 0.0% | 3.1% |
| canonical | heldout_support | test_query_02 | 0.0% | 4.7% |
| invariant | canonical_support | test_query_00 | 0.0% | 3.1% |
| invariant | canonical_support | test_query_01 | 6.2% | 3.9% |
| invariant | canonical_support | test_query_02 | 5.5% | 4.7% |
| invariant | heldout_support | test_query_00 | 0.0% | 3.9% |
| invariant | heldout_support | test_query_01 | 0.8% | 3.1% |
| invariant | heldout_support | test_query_02 | 0.0% | 6.2% |

Paired differences with 95% bootstrap CIs, in percentage points, resampling facts:

| Contrast | Quadrant | Δ EM (pp) | 95% CI (pp) | Facts |
| --- | --- | ---: | --- | ---: |
| augment_minus_canonical | canonical_query/canonical_support | -75.78 | [-82.81, -67.97] | 128 |
| augment_minus_canonical | heldout_query/canonical_support | 4.17 | [2.34, 6.25] | 128 |
| augment_minus_canonical | canonical_query/heldout_support | 0.00 | [0.00, 0.00] | 128 |
| augment_minus_canonical | heldout_query/heldout_support | 0.78 | [0.00, 1.82] | 128 |
| invariant_minus_canonical | canonical_query/canonical_support | -66.41 | [-74.22, -57.81] | 128 |
| invariant_minus_canonical | heldout_query/canonical_support | 3.91 | [2.08, 5.73] | 128 |
| invariant_minus_canonical | canonical_query/heldout_support | 0.00 | [0.00, 0.00] | 128 |
| invariant_minus_canonical | heldout_query/heldout_support | 0.26 | [0.00, 0.78] | 128 |
| invariant_minus_augment | canonical_query/canonical_support | 9.38 | [3.12, 15.62] | 128 |
| invariant_minus_augment | heldout_query/canonical_support | -0.26 | [-0.78, 0.00] | 128 |
| invariant_minus_augment | canonical_query/heldout_support | 0.00 | [0.00, 0.00] | 128 |
| invariant_minus_augment | heldout_query/heldout_support | -0.52 | [-1.30, 0.00] | 128 |

Memory intervention differences within the same checkpoint, in percentage points with fact-clustered 95% CIs:

Uniform guessing over 16 candidates has expected accuracy 6.25%; always outputting one candidate also scores 6.25% on these balanced facts. Therefore, an increase from 0% to roughly 6% between training conditions cannot establish memory generalization by itself. Also examine real retrieval against shuffled values and zero residuals; beating zero residuals without beating shuffled values may only reflect learning to output candidate words. CIs reflect fact sampling, not training-seed variation.

| Condition | Quadrant | Control | Real / control EM | Δ EM (pp) | 95% CI (pp) |
| --- | --- | --- | --- | ---: | --- |
| augment | canonical_query/canonical_support | shuffled | 24.2% / 2.3% | 21.88 | [14.06, 29.69] |
| augment | canonical_query/canonical_support | empty | 24.2% / 0.0% | 24.22 | [17.19, 32.03] |
| augment | heldout_query/canonical_support | shuffled | 4.2% / 4.2% | 0.00 | [0.00, 0.00] |
| augment | heldout_query/canonical_support | empty | 4.2% / 0.0% | 4.17 | [2.34, 6.25] |
| augment | canonical_query/heldout_support | shuffled | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| augment | canonical_query/heldout_support | empty | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| augment | heldout_query/heldout_support | shuffled | 0.8% / 0.0% | 0.78 | [0.00, 1.82] |
| augment | heldout_query/heldout_support | empty | 0.8% / 0.0% | 0.78 | [0.00, 1.82] |
| canonical | canonical_query/canonical_support | shuffled | 100.0% / 0.0% | 100.00 | [100.00, 100.00] |
| canonical | canonical_query/canonical_support | empty | 100.0% / 0.0% | 100.00 | [100.00, 100.00] |
| canonical | heldout_query/canonical_support | shuffled | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| canonical | heldout_query/canonical_support | empty | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| canonical | canonical_query/heldout_support | shuffled | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| canonical | canonical_query/heldout_support | empty | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| canonical | heldout_query/heldout_support | shuffled | 0.0% / 0.3% | -0.26 | [-0.78, 0.00] |
| canonical | heldout_query/heldout_support | empty | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| invariant | canonical_query/canonical_support | shuffled | 33.6% / 5.5% | 28.12 | [19.53, 36.72] |
| invariant | canonical_query/canonical_support | empty | 33.6% / 0.0% | 33.59 | [25.78, 42.19] |
| invariant | heldout_query/canonical_support | shuffled | 3.9% / 3.9% | 0.00 | [-0.78, 0.78] |
| invariant | heldout_query/canonical_support | empty | 3.9% / 0.0% | 3.91 | [2.08, 5.73] |
| invariant | canonical_query/heldout_support | shuffled | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| invariant | canonical_query/heldout_support | empty | 0.0% / 0.0% | 0.00 | [0.00, 0.00] |
| invariant | heldout_query/heldout_support | shuffled | 0.3% / 0.0% | 0.26 | [0.00, 0.78] |
| invariant | heldout_query/heldout_support | empty | 0.3% / 0.0% | 0.26 | [0.00, 0.78] |

Limitations:

- Only completed suites and runs with prediction-level verification enter summaries.
- Held-out query EM is a macro average over XML, CSV, and dialogue templates; CI resamples facts, not views.
- Held-out support is one of three formats per fact, round-robin; it is not all nine query/support combinations per fact.
- Equal optimizer updates/exposures do not imply equal prompt-token counts or FLOPs; paired-view consistency adds work.
- Single-seed confidence intervals exclude training randomness and have no multiplicity correction.
- Finite 16-word synthetic labels and authored format stress tests are not a natural-language benchmark.
- Uniform guessing among 16 labels has expected EM 6.25%; emitting any fixed vocabulary word also scores 6.25% on the balanced confirmation facts. Improvement toward this level need not indicate fact-specific memory use.
- Memory-use evidence compares real reads against both shuffled values and zero-residual reads within the same checkpoint and facts, separately from between-condition training gains; a positive real-minus-empty effect alone may reflect output-vocabulary adaptation.
- The historical failed paraphrase is now a training view, not confirmation evidence.
- Forced-correct-value oracle changes the per-token read distribution and is not a mathematical upper bound.
- Empty means zero memory residual; shuffled values can accidentally preserve the answer category.

Excluded runs or comparisons:

- generalization_smoke_20261005/augment32smoke: Smoke excluded: it uses development facts and templates
