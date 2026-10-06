# Scaling and cold-start experiment summary

Generated at (UTC): 2026-10-05T17:01:59.864463+00:00

List only completed runs that pass prediction-level verification. Incomplete runs or inconsistent evidence are excluded; independent smoke-test entities do not enter the main tables.

EM tables show correct/total facts in controlled synthetic experiments. CIs describe fact-sampling variability conditional on a fixed model, not training randomness.

The primary measure contrasts real per-token retrieval with shuffled and empty controls from the same checkpoint. The forced-correct-value diagnostic (data field oracle) injects one correct value at every token, changing the distribution established by real sparse reading. It is not a mathematical upper bound; real-retrieval EM can exceed it. A low diagnostic score alone cannot identify inadequate readout as the main bottleneck, or establish whether per-token retrieval errors have compensating effects; further ablations are needed.

## Data scale, training budget, and memory performance

LM steps gives the source training update budget; parentheses show new updates in the current run. Selected is the checkpoint step actually chosen on development data.

| Run / group | Variant | N | LM steps (new) | selected | cold train/deploy | pre real | immediate real | final real | Forced-value diagnostic | shuffled | empty |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| scaling_cold_20261005/coldtrained_emptyinit / G1 | stable | 4096 | 512 (0) | 512 | 128/0 | 0/128 (0.0%) | 32/128 (25.0%) | 31/128 (24.2%) | 12/128 (9.4%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_cold_20261005/stable4096cold128 / G1 | stable | 4096 | 512 (512) | 512 | 128/128 | 3/128 (2.3%) | 62/128 (48.4%) | 62/128 (48.4%) | 12/128 (9.4%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_extended_20261005/stable4096long / G1 | stable | 4096 | 1536 (1536) | 1536 | 0/0 | 7/128 (5.5%) | 127/128 (99.2%) | 127/128 (99.2%) | 87/128 (68.0%) | 2/128 (1.6%) | 0/128 (0.0%) |
| scaling_raw_20261005/raw128 / G1 | raw | 128 | 512 (512) | 512 | 0/0 | 0/128 (0.0%) | 1/128 (0.8%) | 0/128 (0.0%) | 118/128 (92.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_raw_20261005/raw4096 / G1 | raw | 4096 | 512 (512) | 384 | 0/0 | 0/128 (0.0%) | 6/128 (4.7%) | 0/128 (0.0%) | 127/128 (99.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable1024 / G1 | stable | 1024 | 512 (512) | 512 | 0/0 | 2/128 (1.6%) | 48/128 (37.5%) | 51/128 (39.8%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable128 / G1 | stable | 128 | 512 (512) | 512 | 0/0 | 0/128 (0.0%) | 13/128 (10.2%) | 17/128 (13.3%) | 8/128 (6.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable4096 / G1 | stable | 4096 | 512 (512) | 512 | 0/0 | 3/128 (2.3%) | 55/128 (43.0%) | 56/128 (43.8%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/standardtrained_coldinit / G1 | stable | 4096 | 512 (0) | 512 | 0/128 | 3/128 (2.3%) | 53/128 (41.4%) | 54/128 (42.2%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |

pre/immediate did not measure Forced-value diagnostic, shuffled, or empty; missing measurements are not zeros.

## Addressing, paraphrases, controls, and value distributions

| Run | final R@1 / R@4 | decode residency / switches per example | paraphrase real / Forced-value diagnostic | control before / after | value effective rank | unique values / records |
| --- | --- | --- | --- | --- | ---: | --- |
| scaling_cold_20261005/coldtrained_emptyinit | 100.0% / 100.0% | 17.6% / 1.59 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 7.38 | 128 / 128 |
| scaling_cold_20261005/stable4096cold128 | 100.0% / 100.0% | 29.8% / 1.54 | 0/128 (0.0%) / 0/128 (0.0%) | 5/64 (7.8%) / 4/64 (6.2%) | 7.77 | 256 / 256 |
| scaling_extended_20261005/stable4096long | 100.0% / 100.0% | 94.4% / 0.45 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 3/64 (4.7%) | 9.79 | 128 / 128 |
| scaling_raw_20261005/raw128 | 40.6% / 72.7% | 4.4% / 1.70 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 6.91 | 128 / 128 |
| scaling_raw_20261005/raw4096 | 50.8% / 88.3% | 7.0% / 2.20 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 5.58 | 128 / 128 |
| scaling_stable_20261005/stable1024 | 100.0% / 100.0% | 24.9% / 1.62 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 1/64 (1.6%) | 8.07 | 128 / 128 |
| scaling_stable_20261005/stable128 | 50.8% / 78.9% | 11.7% / 1.70 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 1/64 (1.6%) | 10.06 | 128 / 128 |
| scaling_stable_20261005/stable4096 | 100.0% / 100.0% | 21.4% / 1.76 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 2/64 (3.1%) | 7.28 | 128 / 128 |
| scaling_stable_20261005/standardtrained_coldinit | 100.0% / 100.0% | 14.2% / 1.87 | 0/128 (0.0%) / 0/128 (0.0%) | 4/64 (6.2%) / 3/64 (4.7%) | 7.39 | 256 / 256 |

R@k uses the final prompt token before actual generation. Decode residency is the fraction of decode queries whose top-4 includes the correct record, weighted by actual query counts; it is not a top-1 proportion or attention weight. Switches count top-1 changes per example, including prefill to first decode. Historical runs without traces show a dash. Controls are unwritten random facts; their EM is not abstention ability. Value effective rank and uniqueness measure numerical diversity, not memory usefulness by themselves.

## Strictly fact-paired differences

Differences are first minus second, in percentage points. ID sets and labels must match exactly; no intersection-only comparisons.

`deployment_cold128_minus_0_same_checkpoint` changes only the initial deployment bank and requires identical checkpoint SHA and source training configuration, while keeping training cold=0/128 conditions separate. Models from different cold-training conditions are not treated as same-weight deployment controls. `stable4096_LMschedule1536_minus_512` compares a 3x LM schedule: oracle and real-retrieval stages scale together, while alignment remains at 400 updates. This is a combined training-budget comparison, not an equal-FLOPs comparison; total compute is not established to be exactly 3x.

| Contrast | first | second | paired N | Δ EM (pp) | 95% percentile CI (pp) |
| --- | --- | --- | ---: | ---: | --- |
| final_real_minus_empty | scaling_cold_20261005/coldtrained_emptyinit | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+17.19, +32.03] |
| final_real_minus_shuffled | scaling_cold_20261005/coldtrained_emptyinit | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+17.19, +32.03] |
| final_real_minus_empty | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/stable4096cold128 | 128 | +48.44 | [+39.84, +57.03] |
| final_real_minus_shuffled | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/stable4096cold128 | 128 | +48.44 | [+39.84, +57.03] |
| final_real_minus_empty | scaling_extended_20261005/stable4096long | scaling_extended_20261005/stable4096long | 128 | +99.22 | [+97.66, +100.00] |
| final_real_minus_shuffled | scaling_extended_20261005/stable4096long | scaling_extended_20261005/stable4096long | 128 | +97.66 | [+94.53, +100.00] |
| final_real_minus_empty | scaling_raw_20261005/raw128 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_shuffled | scaling_raw_20261005/raw128 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_empty | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw4096 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_shuffled | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw4096 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_empty | scaling_stable_20261005/stable1024 | scaling_stable_20261005/stable1024 | 128 | +39.84 | [+31.25, +48.44] |
| final_real_minus_shuffled | scaling_stable_20261005/stable1024 | scaling_stable_20261005/stable1024 | 128 | +39.84 | [+31.25, +48.44] |
| final_real_minus_empty | scaling_stable_20261005/stable128 | scaling_stable_20261005/stable128 | 128 | +13.28 | [+7.81, +19.53] |
| final_real_minus_shuffled | scaling_stable_20261005/stable128 | scaling_stable_20261005/stable128 | 128 | +13.28 | [+7.81, +19.53] |
| final_real_minus_empty | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable4096 | 128 | +43.75 | [+35.16, +52.34] |
| final_real_minus_shuffled | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable4096 | 128 | +43.75 | [+35.16, +52.34] |
| final_real_minus_empty | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/standardtrained_coldinit | 128 | +42.19 | [+33.59, +50.78] |
| final_real_minus_shuffled | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/standardtrained_coldinit | 128 | +42.19 | [+33.59, +50.78] |
| N4096_minus_N128_matched_updates | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| N4096_minus_N128_matched_updates | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable128 | 128 | +30.47 | [+21.88, +39.06] |
| deployment_cold128_minus_0_same_checkpoint | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+16.41, +32.03] |
| deployment_cold128_minus_0_same_checkpoint | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/stable4096 | 128 | -1.56 | [-3.91, +0.00] |
| stable4096_LMschedule1536_minus_512 | scaling_extended_20261005/stable4096long | scaling_stable_20261005/stable4096 | 128 | +55.47 | [+46.09, +64.06] |

Bootstrap: 10,000 resamples, seed 123. No multiple-comparison correction is applied; intervals from one training seed do not establish significance across training randomness.

## Comparability and exclusions

Matching G group IDs indicate the same feature cache, runtime source, and model revision; cross-group differences do not establish causal effects of data scale.

- G1: cache `e1f6c0165e01bbb6f9196fd3795bdab10eb2b8ba88d2be40aa8e4a1d7d935dff`; source `dc4ee157f238fb27aebbbc313d48ad68b3a118dad3ed9f6cfee24126603d11c5`; model `cdbee75f17c01a7cc42f958dc650907174af0554`.

- Not paired scaling_cold_20261005/stable4096cold128 and scaling_stable_20261005/stable128: Cross-run matching failed: training_cold_records.

- Not paired scaling_extended_20261005/stable4096long and scaling_stable_20261005/stable128: Cross-run matching failed: training_lm_updates.

- Not paired scaling_stable_20261005/standardtrained_coldinit and scaling_extended_20261005/stable4096long: Same-checkpoint deployment matching failed: checkpoint_sha256.

- Not paired scaling_extended_20261005/stable4096long and scaling_cold_20261005/stable4096cold128: Training-budget matching failed: training_cold_records.

- Excluded scaling_smoke_20261005/stable32: Smoke run excluded (independent evaluation entities, seed 8042).

This public JSON export contains aggregate results, evidence hashes, and control relationships, but no per-example answers/predictions, internal absolute paths, model weights, or connection settings.
