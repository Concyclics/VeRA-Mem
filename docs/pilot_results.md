# Pilot experiments: the mechanism is implemented, but useful memory is not yet established

> This is the historical small-data report from before scaling; it preserves the original failures and interpretations at the time. Subsequent stable training reached 127/128 on new facts with the same template; see the [scaling results and student handoff](scaling_results.md). That later result still did not solve paraphrase generalization.

Experiment date: 2026-10-05. Base model: **Qwen3-4B-Instruct-2507**, fixed revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Experiments used an NVIDIA H100 PCIe. These are small exploratory experiments, not confirmatory evidence of method superiority or long-term memory.

**Main finding: the per-token VDB–VeRA read/write path works, but the current training recipe has not produced usable new-fact memory.** In the final round, the correct record entered top-4 for 30 of 32 new facts, yet generation with real retrieval scored 0/32; forced correct-value reads scored 3/32. Text retrieval scored 32/32. The next priority is value-to-VeRA readout and its training, before increasing database scale.

![Staged synthetic-memory and MedMCQA retention controls](results/pilot_overview.png)

Each group contains only 32 facts. The two panels use different data and conditions; retrieval hit rate must not be equated with answer accuracy.

## Executed mechanism

An adapter branch is inserted at Qwen's zero-indexed layer 20, `mlp.down_proj`, with input dimension 9728 and output dimension 2560. Every forward pass forms a query from the current token's actual layer input. Online keys/values remain in a CPU exact vector store; cosine top-4 retrieval returns a mixed 64-dimensional value to the GPU:

```text
q_t = normalize(Wq · RMSnorm(x_t))
vbar_t = weighted_top4_values(q_t, VDB)
delta_t = b ⊙ B[(A x_t) ⊙ vbar_t]
y_t = frozen_down_proj(x_t) + delta_t
```

A/B are fixed random matrices. Wq/Wk/Wv/b train offline and are all frozen online. A complete new support observation passes through the backbone with adaptation disabled; the same layer's input at the final prompt token generates its key/value, which is then committed to the VDB. Reads do not update the bank, and each generation uses a fixed snapshot. No shared-parameter gradient update is performed for a new fact online.

This is **input-generated external parameterized vector memory**: values act as dynamic rank-scaling parameters in the VeRA branch. Observation supports contain new-fact answers, so this is writing after supervised observation, not unsupervised TTT. The prototype implements a CPU exact flat index and atomic persistence, not a production concurrent VDB service.

## Data, training budget, and controls

| Item | Setting in this round |
| --- | --- |
| Offline synthetic train / dev | 128 / 32 distinct entities; a balanced vocabulary of 16 common answer words |
| Each online condition | 32 priming observations + 32 new facts + 16 never-written control facts |
| Online measurements | Predict before writing; measure each fact immediately; test old facts every 8 records; finally test priming, controls, and alternative question wording |
| Adapter dimensions | rank=64, key_dim=64, top-k=4, temperature=0.2 |
| Offline training | 200 Q/K contrastive-alignment steps; 3 epochs, 384 answer-supervised updates |
| Optimizer | Adam; Q/K/V learning rate 0.001; b learning rate 0.03; gradient norm capped at 1 |
| LoRA controls | One r4 slot; four r1 slots; four r4 slots; 5 steps per priming/online fact, learning rate 0.0005 |
| Generation/scoring | Official chat template; greedy; at most 8 new tokens; normalized full-answer EM |
| MedMCQA | 64 questions sampled from train and split into priming/online; 16 independent validation questions as controls; explanations are not used |

Training loss covers only answers and EOS. Reported NLL/PPL covers answer tokens only, excluding EOS; PPL is `exp(total_answer_nll / total_answer_tokens)`. MedMCQA additionally reports accuracy from conditional log probabilities of the four candidate letters; in this round its final scores match generated-letter EM.

All three rounds—original, centered, and staged—share training seed 42 and fixed offline entities. Online evaluation seeds are respectively 42, 43, and 44; the latter two use new facts and priming records. This is **an iterative diagnostic development process, not independent replication with three training seeds**. Differences across online seeds cannot directly test a single-factor effect; controls within each round use the same facts.

## Three rounds of main-architecture results

Each generation denominator is 32 online facts. Retrieval is measured at the last prompt token before generation, without teacher-forced answer tokens.

| Run | Online seed | Real retrieval top-1 / top-4 | Real-retrieval EM | Correct-value EM | Permuted-value EM | Zero-residual EM |
| --- | ---: | --- | --- | --- | --- | --- |
| Original joint training | 42 | 9/32 / 21/32 | 0/32 | 0/32 | 0/32 | 0/32 |
| Center only the Wv input | 43 | 0/32 / 2/32 | 1/32 | 0/32 | 0/32 | 0/32 |
| Centering + frozen addressing + correct-evidence readout training | 44 | 18/32 / 30/32 | 0/32 | 3/32 | 0/32 | 0/32 |

Text TF-IDF and text-oracle controls score 32/32 in every round. In the last round, oracle means forcing the correct record's single value. This is a diagnostic condition, **not a mathematical upper bound guaranteed to score highest**: a single value and a real top-4 mixture have different distributions.

### Original failure: every value is identical

The original final VDB contains 64 values but only **1 unique vector**. Every coordinate has standard deviation 0; all 4096 elements satisfy `|v|≥0.999`; every vector has norm 8. Every pair of values has cosine 1, while the keys remain distinct.

Real retrieval, oracle, and permutation therefore collapse to the same parameterized-value modulation. Answer-token NLL falls from 12.477 without memory to approximately 2.76, without improving new-fact EM. A shared branch learning an answer-vocabulary/format prior can explain that reduction; it does not prove facts were stored. See the [original value-distribution diagnostic](results/diagnostics/initial_value_collapse.json).

### Centering prevents identical values, but addressing degenerates

Only the Wv input changes: subtract the mean RMS-normalized feature of **offline training supports**, saved as a fixed checkpoint buffer. Wq/Wk, learning rates, and the remaining architecture are unchanged. The 64 values become 64 distinct vectors and saturation falls to 28.9%. However, mean pairwise key cosine reaches 0.995, while dev top-1 falls from 53.1% after warm-up to 3.1% after joint training.

This is negative evidence about the current joint-optimization recipe: preventing one writer degeneration does not ensure useful addressing or readout. Unique-vector count alone does not establish useful memory. [Centered-vector diagnostic](results/diagnostics/centered_vectors.json)

### Staging preserves addressing, but readout remains weak

The third round freezes Wq/Wk after the same contrastive alignment and clears their old gradients. It trains only Wv/b, supplying the correct support's single value during offline training. Checkpoints are selected by **dev oracle answer-token NLL**, not online-test results.

Dev top-1/top-4 stays at 53.1%/93.75%; the new online stream reaches 56.25%/93.75%. All 64 values are distinct, with approximately 22.5% saturation. Yet real-retrieval EM remains zero, oracle reaches only 3/32, and even oracle scores 0/32 under alternative question wording. The evidence at this stage establishes changing values and functioning addressing, not reliable factual-memory readout. [Staged-vector diagnostic](results/diagnostics/staged_vectors.json)

## LoRA and medical-task controls

The following results use online seed 42. All are same-question retests after observation/writing.

| Method | Trainable adapter parameters | Final synthetic-new-fact EM | Final MedMCQA accuracy |
| --- | ---: | --- | --- |
| Frozen | 0 | 0/32 | 17/32 (53.125%) |
| Text TF-IDF | 0; text index counted separately | 32/32 | 32/32 |
| Text oracle | 0; original text counted separately | 32/32 | 32/32 |
| LoRA 1×r4 | 49,152 | 1/32 | 25/32 (78.125%) |
| LoRA 4×r1 | 49,152 | 3/32 | 24/32 (75%) |
| LoRA 4×r4 | 196,608 | 5/32 | 25/32 (78.125%) |

Only the first two LoRA conditions have equal total parameter counts. Four r4 slots quadruple capacity but do not outperform a single r4 slot on these medical questions. Samples are small, and learning rates/steps were not tuned under matched budgets; this table cannot establish general advantages or disadvantages of multiple slots.

Directly transferring the original synthetic writer to MedMCQA gives 17/32 for real VDB, oracle, and zero residual, with no accuracy improvement. It never learned a writing interface from medical observations. This is therefore only a **synthetic-to-medical cross-domain diagnostic**, not the performance of a sufficiently trained medical-memory system.

The high text-RAG score also has a clear boundary: supports contain legitimately revealed answers, and retests use the same questions/entities. It verifies task and read-path solvability, not 100% generalization accuracy on unseen medical questions.

## Costs, implementation corrections, and evidence boundaries

- The main architecture learns 1,870,336 shared parameters offline. Fixed random A/B contain 786,432 float32 elements, approximately 3 MiB. Online, these parameters are frozen and new facts are written only as vectors.
- Each key/value uses `(64+64)×4 = 512` bytes. Numeric payload for 64 records is 32 KiB, excluding IDs, timestamps, containers, and serialization overhead. The centering-statistics buffer adds 38,912 bytes.
- These short synthetic support texts can be smaller than 512 bytes. This round **does not establish storage savings over text**, or advantages in large-scale retrieval, CPU-offload latency, or total training cost.
- GPU peaks include offline training in the same process, so they are not isolated memory costs for individual reading methods. Repeated conditions may reuse frozen-feature caches; write/baseline-read timing is not a deployment benchmark.
- The initial baseline evaluator omitted the alternative-question flag. Original records remain intact; a zero-training-step `paraphrase_correction` reevaluation used the same checkpoint, and public statistics use the corrected results. This did not affect original-question final EM or the main VDB method's paraphrase evaluation.
- Value permutation breaks entity–vector correspondence, but a finite vocabulary allows some permutations to retain the same answer class. It does not mean every semantic answer becomes wrong.
- Zero residual preserves the same record lifecycle while disabling memory injection. Its outputs equal an empty read, but it is not a zero-storage-cost condition.
- There is only one shared training seed, a fixed support template, and a finite vocabulary. Unknown-entity controls have hidden random-word labels; their EM is not an abstention rate. LongMemEval, LoCoMo, conflict updates, million-record capacity, and multiple-training-seed confirmation were not completed in this round.

Full itemized results and fact-paired bootstrap:

- [Original and MedMCQA](results/initial/report.md) / [machine-readable statistics](results/initial/summary.json)
- [Centering diagnostic](results/centered/report.md) / [machine-readable statistics](results/centered/summary.json)
- [Staged diagnostic](results/staged/report.md) / [machine-readable statistics](results/staged/summary.json)

Paired bootstrap describes variation over this fact sample, not training randomness. The original and later rounds use different data; cross-round differences are not presented as statistically significant improvements.

## Priorities for students

1. **Pass the readout check first.** Fix the correct value and establish fitting on a smaller training set before testing independent entities and alternative templates. Compare correct, permuted, and zero values. Do not start by enlarging the VDB or optimizing NLL alone.
2. **Stabilize the writer.** Retain centering diagnostics; investigate support content tokens/local pooling, auxiliary value reconstruction, or semantic supervision. Continuously record variance, effective rank, and saturation. These changes have not yet shown gains in this round.
3. **Then restore real per-token reading.** Preserve the validated query/key checkpoint and compare top-1/top-4, weight sharpness, and evidence switching during generation. Question-level fixed retrieval is a diagnostic ablation, not a replacement for the main architecture.
4. **Then test continuity.** Add new facts, conflicting upserts, unknown questions, blockwise retention, and capability interference. Separate writing, addressing, readout, and forgetting metrics.
5. **Finally move to real conversations and confirmation.** Fix the training configuration first, then confirm with at least three new training seeds and unused online facts. Use MedMCQA to connect with existing work, and LongMemEval cleaned/LoCoMo for external long-conversation validation.

See [literature_and_design.md](literature_and_design.md) for literature and the full follow-up matrix, and [data_protocol.md](data_protocol.md) for data permissions and splits.
