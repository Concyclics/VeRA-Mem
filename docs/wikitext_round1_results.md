# WikiText-103 first round: corpus adaptation without successful online recall

The complete first round finished on H200 on 2026-10-14. All three VeRA readers and parameter-matched LoRA processed **124,273,491 LM targets** plus **613,400 episodic answer targets**. Full static evaluation and the 1,000-unit continual-memory evaluation completed. The report-export failure was subsequently repaired without changing training, evaluation, or any prediction.

**The result does not establish useful continual learning at this scale.** The memory readers improve static language modeling, but newly written observations do not enter the candidate set at the initial answer position, and counterfactual edits are not recovered. The whole-block operator does not outperform the matched factor-mean operator or LoRA on test perplexity.

Sources: [registered protocol](wikitext_round1_protocol.md), [complete paired analysis](results/wikitext_round1/analysis.json), [execution audit](results/wikitext_round1/execution_audit.json), and [generated report](results/wikitext_round1/report.md). This is one seed (`20261013`); the intervals below quantify held-out article sampling, not training-seed robustness.

## Static language modeling

The frozen Qwen3-4B-Instruct-2507 backbone is identical across methods. Memory contains training observations only. Primary test scores exclude the one train-overlapping article and cover **294,938 targets from 59 articles**. Lower PPL is better.

| Method | Trainable parameters | Validation PPL | Clean test PPL |
| --- | ---: | ---: | ---: |
| Frozen base | 0 | 10.3544 | 9.8387 |
| Diagonal VeRA | 1,118,400 | 9.1651 | 8.7185 |
| Factor-mean outer VeRA | 1,126,656 | 8.6404 | 8.2674 |
| Whole-block outer VeRA | 1,126,656 | 8.6474 | 8.2732 |
| Matched LoRA | 1,130,496 | 7.8388 | 7.5185 |

Whole-block VeRA reduces test PPL by **15.91%** relative to the frozen base. Its paired PPL ratio to the base is 0.8409, with an article-bootstrap 95% interval of [0.8340, 0.8471]. However, its PPL is **10.04% higher than LoRA's**. The block/factor-mean ratio is 1.000708 [1.000190, 1.001262], a small disadvantage for the block arm in this run. These data do not support a block-operator advantage.

Shuffled-value validation still beats the frozen base for all readers (PPL ratios 0.8830, 0.9071, and 0.8698, respectively). This is consistent with substantial generic corpus adaptation. It is not evidence that newly written content is being recalled correctly.

## Frozen-weight continual memory

Confirmation uses 1,000 observed passages from 58 non-overlapping test articles, each queried in two unseen phrasings: **2,000 query instances per condition**. The task is anchored three-word passage completion, not open-domain QA. All shared weights and statistics have identical hashes before and after the stream.

| Reader | Pre-write A EM | Stream A EM | A gain (percentage points, 95% CI) | Edited B EM | Joint A/B/restoration EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal | 0.70% | 0.75% | +0.05 [0.00, 0.18] | 0.00% | 0.00% |
| Factor-mean outer | 0.50% | 0.50% | +0.00 [0.00, 0.00] | 0.00% | 0.00% |
| Whole-block outer | 0.60% | 0.60% | +0.00 [-0.16, 0.14] | 0.00% | 0.00% |

Frozen-base EM is 0.20%; offline-trained, frozen LoRA reaches 1.90%. A gain over the base is insufficient to demonstrate online learning: the appropriate paired pre-write comparison is essentially flat. All edited-B versus stale-A EM gains are zero. Stable first-batch EM under later writes also cannot establish useful retention when acquisition and counterfactual updating have failed.

At the **initial answer position**, all three readers retrieve the newly observed target block in **0 of 2,000 queries**, both for the streamed A observations and edited B observations. This is candidate recall@32, before fine reranking, and is distinct from ANN approximation recall. The measurement does not cover every later generation position.

The correct-block intervention does not rescue the content read: oracle-A EM is 0.80%, 0.35%, and 0.45% for diagonal, factor-mean, and block readers; oracle-B EM and first-three-word EM are zero for all three. Thus, fixing coarse candidate access alone is not demonstrated to be sufficient. The oracle supplies one correct block rather than the ordinary four-block mixture, so it is an intervention with a changed input distribution, not a guaranteed performance upper bound. Ratios that divide by its small positive gain can exceed one and should not be interpreted as successful retrieval.

## Metric and teacher limitations

The contextual frozen teacher reaches only 3.65% strict EM on A and 7.40% on B. Its first-three-word diagnostic reaches 30.05% and 51.90%, respectively, showing that generation format and excess continuation depress strict EM. We retain the registered strict metric rather than replacing it after seeing the results.

Formatting does not explain the entire failure: edited-memory and oracle-B first-three-word EM are also zero. The retrieved-text control, using the block reader's own retrieved passages, reaches 0.05% stream EM. This control shares the failed candidate route; it is not an independent lexical or embedding RAG baseline.

## Index, resources, and persistence

The initial store contains **984,723 blocks**, each with eight low-rank K/V rows. GPU IVF-Flat uses 4,096 centroids and the registered fallback budget `nprobe=1024`. Initial qualification obtained **90.39%** approximate-neighbor recall@32, below the 95% target, at about 17,820 queries/s for the specified 256-query batch. After training, the held-out LM-query ANN check measured approximately 93.0–93.2% recall. These figures measure approximation to exact coarse-vector neighbors, not recall of the semantically relevant newly written passage. They do not establish what exact search would retrieve for the continual-memory prompts.

The two formal training stages took about **17.61 hours** in total; full confirmation took about **59.5 minutes**. The pilot, 10M diagnostic, preprocessing, and GPU waiting time are additional. The recorded four-arm training peak was below 43 GiB of PyTorch allocated memory. Runtime is a single-host observation, not an isolated hardware benchmark.

The 30 stream minibatch commits write A, replace it with B, and restore A. They leave **985,723 active records** and **987,723 physical records**, with superseded versions masked. The writer accepts observed text features; gold QA answers are not its input. The final checkpoint, restored index, new K/V payloads, predictions, logs, code snapshot, and provenance are retained outside Git and mirrored locally. The [execution audit](results/wikitext_round1/execution_audit.json) records the final backup verification.

The report-export bug was a NumPy boolean produced by a derived oracle-gain threshold. Converting the upstream scalar to a Python float fixes serialization. The original failed log and controller state are retained; a separate CPU analysis repair uses unchanged measurements and records its own source hash. A regression test now exercises condition-file joins with a positive oracle gain through JSON export.

## Implications for the next experiment

Do not treat this result as confirmation of the earlier small-data generalization claim, and do not scale the corpus further on this evidence. The implementation runs at the intended corpus size, but the central online-memory claim fails this protocol.

Before another full pass, isolate two issues on development data:

1. **Addressing:** measure the true written block's exact coarse-vector rank for natural and paraphrased memory queries, separately from ANN recall. Compare retrieval objectives or query representations only after confirming that the intended observations can be reached without forced candidates.
2. **Content readout:** require reliable held-out reconstruction and counterfactual B recovery under a controlled correct-block input. Include explicitly edited observed supports during offline write/read training, and qualify contextual-teacher formatting separately. A successful candidate lookup must produce sensitivity to the newly observed values.

Keep the completed test predictions sealed. Subsequent choices require development data and a new confirmation protocol. This round aligns WikiText corpus exposure with the earlier paper-scale plan; it does not reproduce the backbones, parameter counts, multi-epoch budgets, or published scores of Memory Decoder or MLP Memory.
