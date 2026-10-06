# TTT, writable memory, and this project's scaling experiments

Review date: 2026-10-05. This document separates original-paper evidence, interpretations of current failures, and designs awaiting tests. It motivates this round of experiments; **unexecuted scales or ablations are not presented as completed results**. Actual results are established by the corresponding run directories, configurations, and reports.

## 1. Insufficient data is plausible, but not the only explanation

The preceding round used only **128 offline facts, 3 epochs, and 384 answer-supervised updates** to learn approximately 1.87 million shared parameters and teach frozen Qwen3-4B to read a new rank-64 parameterized interface. This is far below the budgets used by published methods to train read/write interfaces. However, identical values in the first version, key cosines concentrating at 0.995 after joint training, and only 3/32 correct even with the correct value separately indicate optimization degeneration, addressing instability, and inadequate readout. More examples do not guarantee a fix. [Original project evidence](pilot_results.md)

Test two questions together: **Do more independent facts improve generalization, and do better initialization and a better training interface make those facts learnable?** Repeating the same data for more epochs, or adding records to a VDB that has not learned to read them, cannot answer the first question.

## 2. What the original papers actually trained

Offline training volume, test-stream length, and number of test questions have different units. Small evaluations after large-scale offline training are not evidence that a few training examples suffice to create memory.

| Work and original source | Offline training and initialization | Test-time update | Most relevant limits for this project |
| --- | --- | --- | --- |
| [TTT, v4, §2.7, Table 3, Appendix C](https://arxiv.org/html/2407.04620v4) | Networks containing TTT layers are trained from scratch. Models of 125M / 350M / 760M / 1.3B use 2.5B / 7B / 15B / 26B tokens and 4,800 / 13,500 / 29,000 / 50,000 outer-loop steps. Approximately 0.5M tokens per batch. Shared initial state `W0` is learned in the outer loop and reportedly improves stability. | Each sequence starts from the shared initial state and performs self-supervised inner-loop updates on projected inputs, in minibatches of 16 tokens. Inner-loop base LR is 1 for Linear and 0.1 for MLP, multiplied by an input-dependent sigmoid gate. | Supports a learned initial state. Those LRs belong to a specific reconstruction inner loop, not this project's Adam writer. Main evaluations are Pile/Books LM with context 1k–32k, not an add-on to frozen Qwen. |
| [Titans, v1, §3, §§5.1–5.4](https://arxiv.org/html/2501.00663v1) | Neural memory is part of the initial architecture and trained end to end. Models of 170M / 340M / 400M each use 15B FineWeb-Edu tokens; 760M uses 30B. Training length 4k, batch 0.5M tokens, AdamW LR 4e-4, weight decay 0.1. Persistent memory is separate from input-dependent fast memory. | Neural memory uses key–value reconstruction gradients, momentum, forgetting, and input-dependent rates. The outer loop learns projections and the overall network. | Supports a stable fast/slow-state division, not immediate readability of a random add-on. RULER S-NIAH at 2k/4k/8k/16k and long BABILong streams are different tests; “over 2M context” is not the offline sample count. |
| [M+, v2, §§3.2.3–3.2.4, 4.3](https://arxiv.org/html/2502.00592v2) | Based on Llama-3.1-8B, first trains the MemoryLLM interface for **1,200,000 steps / 4 weeks**. Stage two takes 200,000 documents from each of four 4k–64k length buckets, mixes equally with FineWeb, and runs 1 epoch. Stage three trains long-memory reading on a separate set of documents. | New text produces latent memory; vectors evicted from short memory enter CPU long memory and are sparsely retrieved by jointly trained query/key projections. | This is not training only an unadapted readout layer. Retention tests take the first 100 filtered questions each from SQuAD/NaturalQA and insert irrelevant contexts; 100 is the evaluation-question count. The paper does not provide enough detail to reconstruct all stage LRs, so missing values must not be guessed. |
| [Larimar, v4, §§2.1–2.2, 5](https://arxiv.org/html/2403.11901v4) | After initialization from pretrained encoder/decoder models, **encoder, memory, and decoder train jointly**. Uses 7.6M samples of 64 tokens; the 6B version runs 10 epochs, Adam LR 5e-6, batch 32, on 8×A100. Trained memory becomes prior `M0`; losses also include memory-free reconstruction and base language-modeling retention. | One-shot writing and sequential updates use least squares/pseudoinverses, without retraining the model by gradients for each new fact. | Joint prior/interface training is highly relevant, but the decoder is not always frozen. CounterFact single-fact tests use the first 2,000 cases; edit counts must not be conflated with 7.6M offline samples. |
| [Doc-to-LoRA, v1, §3, Appendices A/B](https://arxiv.org/html/2602.15902v1) | Freezes the target LLM and trains a hypernetwork mapping document activations to LoRA. Simplified NIAH: 640k examples, 32–256 tokens, 1 epoch, LR 4e-5, approximately 3h on one H200. Full training uses about 3.2M independent contexts, first 80k single-chunk steps then 20k composition steps, accumulating over 200k context tokens per batch across 8 GPUs. | Parameters for a new document are generated by a forward pass; the hypernetwork is not retrained for that document. | The closest scale reference for learning a new parameter interface with a frozen backbone. The authors learn single-chunk readout before composition. Simplified NIAH uses the same question format in train/test and therefore does not establish question-format generalization. |

These are settings reported by the papers, **not minimum budgets required for this project**, and token counts cannot rank sample complexity across different architectures.

Additional code checks:

- The [official TTT repository](https://github.com/test-time-training/ttt-lm-jax) supplies training/data-preparation scripts. Outer-loop peak LRs by model scale are 3e-3 / 1.5e-3 / 1.25e-3 / 1e-3; do not confuse them with inner-loop LRs.
- The [official M+ repository](https://github.com/wangyu-ustc/MemoryLLM) explicitly places M+ training code on a separate branch. Old MemoryLLM YAML files on the main branch do not fully configure all M+ stages.
- The [official Larimar example YAML](https://raw.githubusercontent.com/IBM/larimar/main/larimar_base/configs/config_train_larimar.yaml) uses another setting: BERT-base/GPT-2, LR 5e-5, batch 64, 4 epochs. Do not combine it with the paper's 6B configuration as though they were one experiment.
- The [official D2L implementation](https://github.com/SakanaAI/Doc-to-LoRA) releases an already trained hypernetwork. Its weights cannot directly initialize Qwen3-4B.

## 3. Split cold start into two testable mechanisms

**Learned prior slots.** Optimize a set of keys/values and VeRA readout parameters in offline training episodes, then clone the same fixed initial state at test time. This supplies usable activation distributions and a general readout prior, not facts about new test entities. Learned slots are shared parameters; their bytes and training cost count toward the budget.

**A bootstrapped factual bank.** Before the test stream begins, pass supports permitted by the protocol through a frozen writer to produce keys/values, for example 256 priming observations. Then introduce entirely new entities. This supplies prior knowledge, as in the user's proposed VDB prefill, but must not contain future test answers, future update versions, or test-generated outputs.

Both mechanisms are reasonable but measure different things. The first primarily tests initialization/joint training; the second tests new writes and interference in a bank with history. High priming accuracy alone does not establish continual learning. Report accuracy before/after new writes, old-fact retention, and facts never written.

Suggested minimal controls:

| Initial state | Shared reader/writer | Legitimate initial facts | Question |
| --- | --- | --- | --- |
| Empty | Offline-trained | 0 | New-writing ability without priming |
| Random | Same checkpoint | Random nonsemantic vectors matched to the cold bank in slot count and norm | Whether more vectors or a nonzero residual alone change behavior |
| Cold | Same checkpoint | 256 disjoint priming supports | Whether legitimate prefill improves stability or merely adds interference |
| Learned prior + Cold | Initial slots and interface jointly trained | The same 256 supports | Whether a learned initial state outperforms merely adding records |

The main comparison need not retrain all four conditions immediately. First compare Empty/Random/Cold with the same checkpoint. If a learned prior changes the architecture, give it a separate matched-data/step training control. Gains from longer training must not be attributed to cold start.

## 4. Feasible data scales and budgets for this round

Use nested offline entity subsets `train_128 ⊂ train_1024 ⊂ train_4096`. Keep dev, priming, online new facts, unobserved facts, and alternative-template tests identical across conditions, with entities disjoint from training. The generator produces manifests, IDs, and hashes before training starts.

| Condition | Distinct training facts | Mean exposures per fact | Total training-episode exposures | Purpose |
| --- | ---: | ---: | ---: | --- |
| N128-E3 | 128 | 3 | 384 | Repeat the old budget to isolate structural changes |
| N1024-E3 | 1,024 | 3 | 3,072 | Medium-scale expansion |
| N4096-E3 | 4,096 | 3 | 12,288 | 32 times as many independent facts |
| N128-matched | 128 | 96 | 12,288 | Same total exposures, repeatedly visiting a small set |
| N1024-matched | 1,024 | 12 | 12,288 | Effect of independent data at fixed exposures |

These are episode exposures, not necessarily optimizer steps. With effective batch B, ideal updates are approximately exposures/B. Actual logged `optimizer_steps`, effective answer tokens, and total forward/backward tokens are authoritative. Fixed-epoch curves change both data and compute; matched groups help separate them. If truncation length changes, report training tokens and GPU time as well.

Start with paired diagnostics at 128 and 1,024 if needed, then run 4,096. Retain 4,096 as an explicit scale experiment instead of staying at 128 indefinitely after negative small-data results. Do not wait until a million facts to check readout.

Suggested shared dev/test sizes: 256 dev facts; 256 priming facts, 256 online new writes, and 128 never-observed facts; one final test each with the original and alternative question formats. If compute requires smaller sets, reduce all methods equally and disclose the change. Begin with one training seed for mechanism exploration, then confirm the fixed recipe with three independent training seeds. Three stream seeds for one model are not three independent training replications.

## 5. Make values informative before relying on sparse retrieval

The following are **engineering hypotheses for this project**, not established VeRA improvements from the cited papers.

1. **Normalize writer inputs and remove easily saturated unconstrained paths.** Compare `tanh(Wv h)` with linear or RMS-normalized values after training-statistics centering/scale normalization. Give the residual a separate small learnable scale. Norm constraints limit amplitude; they do not replace semantic training. Fit statistics only on training supports and save them in the checkpoint.
2. **Allow both kinds of gradients at initialization.** A small residual can protect the backbone, but zeroing writer outputs, readout gates, and projections simultaneously prevents writer gradients. Check writer, reader, and query/key gradient norms in every smoke test.
3. **Establish readout and addressing in stages.** First train readout with the correct support's single value, potentially adding value reconstruction/distillation, until it beats permuted and zero values. Then introduce retrievable distractors and replace oracle reads with actual top-k. Final inference must still derive queries from each token's actual VeRA input; neither an oracle nor an external parser may replace the main method.
4. **Maintain a separate discriminative addressing signal.** Use paraphrases of the same entity as positives and different entities with the same answer/template as hard negatives to prevent answer/template clustering. Contrastively pretrain Q/K, then jointly train at a small LR or temporarily freeze them. Monitor retrieval, not only LM loss. Rebuild banks after offline key-encoder updates; new queries must not search keys in an obsolete coordinate version.
5. **Joint training needs varied episodes.** Write supports and read with different questions within each episode. Randomize banks, order, and old/new record proportions. Giving each training entity a permanently optimized value and fitting the training set does not establish writer generalization.
6. **Protect base-model capabilities.** Train and validate with empty or irrelevant memory. An interface that only emits 16 common words can lower answer NLL while harming general generation. Random-entity experiments with a shared finite vocabulary establish only controlled association learning; later tests must expand answer vocabulary, multitoken values, and relation types.

Initially fix the backbone, insertion layer, and rank. Changing layer count, rank, token selection, value normalization, and sample count at once prevents attribution. Necessary combined fixes may be compared as a new recipe against the old one, followed by component ablations; do not claim every component is independently effective.

## 6. Separate four evaluation bottlenecks

| Dimension | Required metrics | Diagnostic meaning |
| --- | --- | --- |
| Write representation | Unique-value count, per-coordinate std, effective rank, mean norm, pairwise cosine, maximum absolute value; saturation rate when using tanh | Distinct values are not necessarily useful, but identical or saturated values can directly explain failure. |
| Addressing | Actual last-prompt-token Recall@1/@k, MRR, top-k weight entropy, evidence switching during generation; progressively larger banks | A correct record inside top-k may still be diluted by mixing. |
| Readout | Free-generation EM and answer NLL for correct values, real retrieval, permuted values, and zero residual; finite-vocabulary candidate scoring only as a diagnostic | Oracle failure prioritizes reader diagnosis; oracle success with retrieval failure prioritizes addressing/mixing. |
| Continuity | Accuracy before writing, immediately after, and after another 32/128/256 writes; priming retention, conflicting updates, unknown entities, and base capabilities | Separate same-question retest gains from long-term retention. |

Each fact follows `query-before-write → observation → encode/write → query-after-write`. Generation only reads a fixed snapshot. Future supports must not appear among candidate values, contrastive-negative metadata, or query-encoder inputs. Supports for old/new facts that have already appeared on the timeline may contain answers: those are legitimate observations, not evaluation-label feedback.

Reuse facts/checkpoints across correct/permuted/zero-value conditions and save individual predictions for fact-paired bootstrap. Lower NLL alone is insufficient. Minimum evidence should include correct-value EM above permutation/empty-read controls, improved real retrieval, and retained effects for unseen entities/questions. Negative results remain useful if they locate data, optimization, addressing, or readout limits.

The online main path remains **observations generating parameterized values written into a VDB**, with the shared interface frozen. Borrowing TTT initialization and outer-loop training does not turn append/upsert into gradient-based TTT. If online fast-weight gradients are added later, report the mode, learning rate, optimizer-state bytes, and forgetting controls separately.
