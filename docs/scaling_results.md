# VeRA-Mem data scaling, stable training, and cold start: student handoff

Subsequent update: an [independent generalization comparison using data augmentation and consistency training](generalization_results.md) has been completed. This page preserves the historical scaling/cold-start protocol and results; its statistics are not combined with the newer protocol.

Experiment date: 2026-10-05. The base model is a fixed revision of Qwen3-4B-Instruct-2507; training and inference ran on an H100. This page covers the completed scaling experiments. Detailed numbers, strictly paired intervals, and source hashes are in the [full report](results/scaling/report.md), with machine-readable results in [summary.json](results/scaling/summary.json). The previous failures remain documented in the [pilot report](pilot_results.md).

## Current conclusions

**The user-defined loop of VDB vectors serving as parameterized memory works on the controlled new-fact task with a shared template.** The stable version used 4,096 offline training facts and 1,536 batch-8 LM updates. For 128 unseen entities, it asked first, then received observations and wrote them to the VDB, ultimately answering **127/128 (99.2%)** correctly. Permuting only the values in the same model/database reduced this to **2/128**; disabling the memory residual gave **0/128**. Neither the backbone nor shared parameters were updated online.

**Question-paraphrase generalization has not been achieved.** Every configuration scored **0/128** on the single alternative question template. The results establish a read/write mechanism only for this controlled task; they do not establish general conversational memory, open-domain knowledge learning, or a solved long-term continual-learning system.

![Data scaling, training budget and initialized-bank controls](results/scaling/scaling_ablations.png)

## Implemented mechanism

The insertion point is Qwen's zero-indexed `layer=20`, `mlp.down_proj`, with input dimension 9728 and output dimension 2560. VeRA rank and key dimension are both 64. For **the actual layer input of every prompt/decode token**, `x_t`, the model computes:

```text
q_t = normalize(Wq · normalize_query(x_t))
I_t = top4_cosine(q_t, VDB.keys)
v_t = Σ softmax(score[I_t] / temperature)_i · VDB.values_i
y_t = W x_t + b ⊙ B[(A x_t) ⊙ v_t]
```

The retrieved `v_t` is therefore VeRA's dynamic rank-scaling parameter, with conditional effective matrix `W + diag(b) B diag(v_t) A`. The VDB does not return text to the prompt, and reads do not use the correct record ID. The CPU reference store computes cosine against every record and then sparsely mixes the top-4 values. Query cost still grows with record count; this is not a validated large-scale ANN service.

A complete new observation first passes through the frozen backbone to obtain the same layer's input `h`. The system then generates and persists:

```text
key_new   = normalize(Wk · normalize_support(h))
value_new = RMSNormalize(Wv · normalize_support(h))
VDB.upsert(record_id, timestamp, key_new, value_new)
```

The database snapshot stays unchanged during a read. Writing occurs only after the observation is revealed. Record IDs are used only for maintenance and evaluation audit. Feature extraction disables the memory residual; because adaptation is confined to one layer, that layer's input does not depend on its own adapter. Offline training learns approximately 1.87 million shared `Wq/Wk/Wv/b` parameters; Qwen and random `A/B` remain fixed. Online forward-pass vector writes change subsequent conditional parameters, but are not inference-time SGD/TTT.

See the [vector-memory contract](vector_memory_contract.md) for the complete interface, timing, and training/deployment states.

## Why more data or a prefilled bank alone is insufficient

A common direction accounted for about 99.54% of the old normalized support-feature energy: using the [saved center norm](results/diagnostics/centered_vectors.json), `98.4022293`, and the squared norm `9728` of each RMS-normalized input gives `98.4022293² / 9728 ≈ 0.995374`. Previously, `tanh` compressed all values into the same vector. The stable version separately centers query/support features using fixed training-set means and then reapplies RMS normalization; values use a linear projection followed by RMSNorm. Statistics are never fitted on dev/test data or updated online. Individual coordinates may exceed 1, so `abs(value)>0.999` can no longer be interpreted as saturation.

Training was also changed: first align query/key, then train the reader with correct values, then jointly train with real top-4 retrieval, including an addressing loss at answer-prediction positions. The learning rate for shared b decreased from 0.03 to 0.005 and that for the writer from 0.001 to 0.0001; the addressing learning rate during real retrieval is 0.00001. Temperature changed from 0.2 to 0.05. **This is a combined recipe change; its effect cannot be attributed solely to removing tanh.**

An independent CPU linear-probe diagnostic also found that support features trained on 128 records decoded 63 of 64 dev answers; with 4,096 training records, they decoded 64/64. Query features decoded only 4/64 and 6/64 respectively. The observation features already contain answer information, so the main difficulty cannot simply be blamed on the backbone failing to encode the answer. This additionally supervised classifier is an information diagnostic; its score is not a VDB–VeRA score. [Probe record](results/scaling_diagnostics/feature_probe.json)

## Data and fairness

The training sets are strictly nested, answer-balanced collections of 128/1024/4096 random entity–word associations, using 16 common answer words. There are 64 dev facts, 128 online new facts, and 64 unobserved controls, with fully disjoint entities. Smoke tests use separate seeds and entities and do not participate in result selection. All formal configurations share test entities, answers, model, source code, and feature caches.

- Main scaling matrix: 512 batch-8 LM updates, totaling 4,096 target-fact exposures. The small set repeats 32 times, the medium set 4 times, and the large set once.
- Address warm-up is counted separately: 400 updates of 128 pairs, totaling 51,200 pair exposures. Reporting only LM updates would hide this stage.
- Stable curriculum: the first 256 LM updates force the correct value; the next 256 use real retrieval. The old recipe forces the correct value throughout, with addressing frozen.
- Extended training: the same 4,096 facts, 1,536 LM updates, with 768 in each stage and 12,288 target exposures. Address warm-up remains 400 updates. This tests extra training budget, not equal FLOPs.
- Checkpoints are selected only by dev NLL under real retrieval. Every run completes its full update budget, although the selected deployment step may be earlier; the full table records the selected step explicitly.
- Final predictions are greedy, freely generated full answers, evaluated by EM with at most 4 new tokens. Predictions and gold-token NLL are recorded separately; teacher forcing for the latter does not enter the retrieval trace.

In the scaling table, each test fact undergoes `query-before-write → observation → encode/write → query-after-write`. Final recall, paraphrases, and controls are then evaluated after all 128 writes. Raw predictions, configurations, checkpoints, initial/final VDBs, actual exit codes, and source snapshots are preserved.

To avoid repeatedly running the frozen backbone, this round caches independent query/support layer inputs in advance. Future support caches never enter training statistics, earlier queries, or the database; they reach the writer and VDB only at the corresponding observation-reveal boundary. This validates causal dataflow, but cache preparation is excluded from online write latency, so it does not support a production write-performance claim.

## Data-scaling and training-budget results

Every entry below is final recall on 128 new facts. “Forced correct value” is an auxiliary diagnostic, **not a mathematical upper bound**. For a model trained with real retrieval, forcing the same correct value into every token changes the read distribution and can perform worse than the real path.

| Recipe | Training facts | LM update budget | Real retrieval | Forced correct value | Permuted values | Zero residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old centered/oracle recipe | 128 | 512 | 0/128 | 118/128 | 0/128 | 0/128 |
| Old centered/oracle recipe | 4096 | 512 | 0/128 | 127/128 | 0/128 | 0/128 |
| Stable recipe | 128 | 512 | 17/128 | 8/128 | 0/128 | 0/128 |
| Stable recipe | 1024 | 512 | 51/128 | 15/128 | 0/128 | 0/128 |
| Stable recipe | 4096 | 512 | 56/128 | 15/128 | 0/128 | 0/128 |
| Stable recipe, extended training | 4096 | 1536 | **127/128** | 87/128 | 2/128 | 0/128 |

This refines the previous interpretation: 128 facts do not imply that the reader lacks all expressive capacity. With more exposures, the old recipe's correct-value diagnostic reached 118/128. Yet its real retrieval remained at zero, showing that data/budget increases alone did not fix its read-distribution mismatch.

At the same LM update budget, the stable recipe improved from 17/128 to 56/128, a paired difference of **+30.47 percentage points**, with a fact-bootstrap 95% interval of **[21.88, 39.06]**. The further gain from 1024 to 4096 facts was smaller. Extending training on the same data reached 127/128, supporting the need to optimize the interface sufficiently rather than requiring millions of facts before any effect can appear.

With 4096 facts, 1536 versus 512 LM updates gave a paired difference of **+55.47 pp, 95% CI [46.09, 64.06]**. In the extended run, real versus permuted values gave **+97.66 pp, [94.53, 100.00]**.

These are mechanism explorations with one training seed. Bootstrap intervals describe sampling variation over these facts for a fixed model; they do not cover training randomness, templates, or distribution shifts to open-ended tasks.

## Cold start and joint training must be interpreted together

The initial bank contains **128 offline training observations encoded by the learned writer**, with no future test facts. Conditions trained with this initial bank include the same background records in each real-retrieval episode, recompute their keys/values, and deduplicate by record ID. This is initialization from learned factual records, not a table of independently optimized prototype parameters.

Keeping 4096 facts and 512 LM updates fixed, the comparison varies whether training and deployment include this bank. Each row uses exactly the same checkpoint:

| Background bank during training | Empty deployment bank | Deployment bank initialized with 128 records | Deployment difference with identical weights |
| --- | ---: | ---: | --- |
| No fixed initial bank | 56/128 | 54/128 | −1.56 pp, 95% CI [−3.91, 0.00] |
| Fixed initial bank of 128 records | 31/128 | 62/128 | +24.22 pp, 95% CI [16.41, 32.03] |

**Ad hoc prefilling did not yield a general benefit; a model trained with a bank can depend on that initialization distribution.** Comparing 62/128 for the bank-trained recipe with 56/128 for the no-bank recipe changes both training episodes and deployment conditions. Those 6 additional correct answers cannot all be attributed to deployment initialization, and this comparison is not budget-matched to the 1536-update extended run.

Initial and newly added facts store only float32 key64/value64: 512 bytes of numeric payload per record, or 64 KiB for 128 records and 128 KiB for 256. This excludes IDs, metadata, the shared writer, and random projections. The small-bank experiment does not validate million-record retrieval, deletion, conflict resolution, or unbounded capacity.

## Remaining problems and the next student tasks

After extended training, same-template prefill Recall@1/@4 are both 100%; the correct record appears in decode top-4 94.35% of the time. All 128 values are distinct, with effective rank about 9.79. Numerical collapse and same-template decode addressing have therefore improved substantially. However, both real retrieval and forced correct values still score 0/128 on the alternative question template, and real prefill top-4 hits only 9/128.

Correlation between decode residency and accuracy is not one-way causation: an incorrect generated token also changes later queries. Switching counts alone cannot identify why the first answer token was wrong. Correct-value diagnostics, position-specific reliability gates, and controlled injection ablations remain useful.

Students should proceed in this order, using new confirmation entities and at least three independent training seeds for new results:

1. **Address template generalization first.** Train with multiple query/support templates and reserve unseen templates for dev/test. Retain actual layer-input queries, same-checkpoint value permutation/zero-residual controls, and the strict read-before-write protocol. The current single paraphrase template must not be repeatedly reused as an unbiased tuning test.
2. **Then test capacity and updates.** Expand the answer vocabulary and introduce multitoken answers, multiple relations, distracting observations, and temporal overwrites. Record old-fact retention, per-write/query cost, and abstention. Upserting an existing ID does not automatically solve textual entity disambiguation or contradiction detection.
3. **Separate the combined recipe.** At fixed data/budget, individually test centering, RMS values, addressing learning rate, temperature, the oracle-to-real curriculum, and position-specific memory gates. Current results do not establish each component's independent benefit.
4. **Finally validate on real external tasks.** Follow the existing [data protocol](data_protocol.md) for complete-session LongMemEval/LoCoMo splits, with MedMCQA as a separate domain task. Unobserved-entity controls are not general-capability retention tests; evaluate ordinary tasks with empty or irrelevant memory as well.

The initialization/curriculum rationale is documented in [offline budgets and initial states in TTT and related methods](ttt_scaling_review.md) and the [official Engram / Qwen3.8-Flash-Next initialization review](hash_initialization_review.md). Hash-table addressing differs from this project's semantic queries. These sources motivate designs; they do not replace project-specific validation.

## Reproduction and artifacts

Implementation version: code commit `0012cce`; each suite's actual source snapshot and file SHAs are authoritative. Base-model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`. See [tested_environment.json](tested_environment.json) for the environment. All **88 tests passed**, including actual hooks, gradients, data isolation, causal writes, retrieval traces, and statistical pairing.

```bash
# See README for fixed-feature preparation; check GPU availability and replace the UUID.
python scripts/run_scaling_suite.py --workspace .. --gpu "$GPU_UUID" \
  --name reproduction_extended --profile extended \
  --cache ../data/scaling/features_v1.pt

# Aggregate downloaded run records without loading the large model.
python scripts/summarize_scaling.py --runs-root ../runs/xtrah100 \
  --output docs/results/scaling
python scripts/plot_scaling.py --input docs/results/scaling/summary.json \
  --output-prefix docs/results/scaling/scaling_ablations
```

Nine formal training/reevaluation runs were completed: seven independent offline trainings and two deployment reevaluations of the same checkpoints, plus separate smoke tests and fixed-feature preparation. Full raw results are retained in the local project backup. The public repository contains code, research documentation, and aggregate results, not the base model, device connection settings, or original student attachments.
