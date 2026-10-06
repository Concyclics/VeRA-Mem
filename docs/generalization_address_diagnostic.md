# Removing training-template directions: an addressing-only development diagnostic

Projecting out training-template means improves addressing for original queries and new observation formats, but **does not solve generalization to unseen query templates**. Against a canonical observation bank, Recall@1 for the two unseen query templates rises only from 1/64 and 3/64 to 2/64 and 4/64. This motivates further work on representations and template coverage, rather than establishing end-to-end memory generalization.

The diagnostic ran at 2026-10-05 17:52 UTC. The inspectable [complete metrics](results/generalization/address_probe.json) and [probe script](../scripts/probe_generalization_address.py) are preserved together. Raw feature caches and probe weights remain in the local workspace backup and are not uploaded to the repository.

## Controlled protocol

- The backbone is frozen `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Features are the final prompt-token input to `mlp.down_proj` at zero-indexed layer 20, with dimension 9,728.
- Training uses 4,096 entities, 8 query templates, and 4 observation templates; development uses 64 independent entities, with the original template and two templates absent from training. Full definitions are in [augmentation_data.py](../src/vera_mem/augmentation_data.py). Confirmation data enter no statistic fitting, scoring, or selection.
- Both arms train 64-dimensional Q/K encoders initialized from the actual `StableVectorVeRA` seed 42. Adam learning rate is fixed at `1e-4`, with 400 steps, 128 distinct entities per step, bidirectional InfoNCE temperature 0.1, and gradient clipping at 1.0. Each arm receives 51,200 training-pair exposures.
- Entity and query/support-template sampling use independent RNGs matched across arms. Query and support expressions are selected independently for each entity. Views of the same entity are not negatives for one another.
- Read only the fixed step-400 result; do not select checkpoints, projection rank, or hyperparameters on development data. Projection changes only input preprocessing; the remaining training budget is identical.
- **This is a 400-step addressing-pretraining comparison, not the main experiment's 1,536-step LM result.** The probe has no value writes, VeRA residual, generated outputs, or token-wise decode retrieval. Recall is not answer EM.

## Projection and the energy denominator

Process query and support domains separately. For training entity \(i\) and template \(t\), normalize features as

\[
u_{it}=\sqrt{d}\,x_{it}/\|x_{it}\|_2,\qquad
\mu_t=\frac1N\sum_i u_{it},\qquad
\mu=\frac1T\sum_t\mu_t.
\]

Apply SVD to the template-mean difference matrix \([\mu_t-\mu]_t\), collecting nonzero directions as orthonormal columns of \(U\). The relative singular-value threshold is fixed at `1e-5`, with rank capped at the number of templates minus one. Actual ranks are 7 for queries and 3 for supports. The ordinary arm uses \(u-\mu\); the projected arm uses \((I-UU^\top)(u-\mu)\). Each resulting vector is then RMS-normalized with epsilon `1e-6`. All means and directions are estimated from training features only; inference needs no template label.

The common denominator for both energy columns below is **total squared energy across all training entities and templates in that domain, after subtracting the global training mean**:

\[
E=\sum_{i,t}\|u_{it}-\mu\|_2^2.
\]

Template-mean energy is \(N\sum_t\|\mu_t-\mu\|_2^2/E\). Energy removed by projection is \(\sum_{i,t}\|U^\top(u_{it}-\mu)\|_2^2/E\). The latter also removes entity variation in the same subspace and is therefore slightly larger.

| Domain | Template-mean energy / E | Removed energy / E | Projection rank |
| --- | ---: | ---: | ---: |
| Query | 94.06% | 95.29% | 7 |
| Support | 79.94% | 81.51% | 3 |

These are neither variance fractions of raw uncentered activations nor proportions of semantic information. Removing high-energy directions may also remove useful entity information. This probe tests only the effect of that operation on the current addressing task.

## Development results

Every cell has **64 facts** as its denominator. A record card is a multiline field/value format; a leading-constraint format places answer or record requirements first. Counts measure retrieval of the correct fact record, not the answer-word class. Expected random-record Recall@1 is 1/64 and Recall@4 is 4/64.

| Query format | Observation-bank format | Ordinary R@1 | Projected R@1 | Ordinary R@4 | Projected R@4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Original template | Original template | 47/64 | 64/64 | 63/64 | 64/64 |
| Original template | Unseen record card | 13/64 | 30/64 | 28/64 | 54/64 |
| Original template | Unseen leading constraint | 6/64 | 27/64 | 20/64 | 48/64 |
| Unseen record card | Original template | 1/64 | 2/64 | 6/64 | 15/64 |
| Unseen record card | Unseen record card | 2/64 | 3/64 | 7/64 | 7/64 |
| Unseen record card | Unseen leading constraint | 1/64 | 4/64 | 4/64 | 10/64 |
| Unseen leading constraint | Original template | 3/64 | 4/64 | 10/64 | 10/64 |
| Unseen leading constraint | Unseen record card | 2/64 | 4/64 | 5/64 | 6/64 |
| Unseen leading constraint | Unseen leading constraint | 1/64 | 3/64 | 4/64 | 5/64 |

For original queries with new observations, R@1 rises from 20.31% / 9.38% to 46.88% / 42.19%, showing that training-template directions interfere with some addressing. R@1 for new queries against original observations still reaches only 3.13% / 6.25%; when both formats are unseen, performance is around 3–4/64. This projection does not resolve the main generalization bottleneck.

One untested explanation is that the final prompt-position layer input contains strong formatting and response-style components. Training-template directions cover only part of them, leaving unseen query formats in uncovered directions. With only 64 development entities and one initialization seed, this diagnostic makes no significance or generality claim and does not revise the frozen confirmation experiment.

## Provenance and reproduction

The summary JSON stores the model version, cache SHA256, data-protocol fingerprint, probe/module source SHA256 values, training logs, and metrics for all 9 development combinations. Cache SHA256 is `00b94acb30450f31285c5268bb5709485a185c81327cf0dabcf02c1e1c66ab32`. Only aggregates are published, not individual entity IDs, predictions, raw caches, or probe weights.

Run in a local environment with the same cache:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python scripts/probe_generalization_address.py \
  --cache /path/to/generalization/features_v1.pt \
  --output /path/to/private/probe/metrics.json
```

By default, the script uses CPU with 4 PyTorch threads and refuses to overwrite existing results. Environment versions are recorded in the aggregate JSON.
