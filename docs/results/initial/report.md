# Preliminary memory experiment results

Suite status: **complete**.

Only aggregate measurements are published here. A partial suite is not evidence that unfinished methods failed.

## Final observed-stream performance

For a partial method, the final phase is the greatest available block number; it may be earlier than the requested end of the stream.

| Run | Dataset | Method | Training / evaluation seed | State | Phase | Correct / n | EM | Choice accuracy | Paraphrase EM | Answer-token NLL | Answer-token PPL | Retrieval @1 / @k |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_medmcqa | medmcqa | frozen | 42 / 42 | complete | block_32 | 17 / 32 | 53.12% | 53.12% | — | 6.1618 | 474.2658 | — / — |
| baseline_medmcqa | medmcqa | lora1_r4 | 42 / 42 | complete | block_32 | 25 / 32 | 78.12% | 78.12% | — | 0.7701 | 2.1600 | — / — |
| baseline_medmcqa | medmcqa | lora4_r1 | 42 / 42 | complete | block_32 | 24 / 32 | 75.00% | 75.00% | — | 0.9289 | 2.5318 | — / — |
| baseline_medmcqa | medmcqa | lora4_r4 | 42 / 42 | complete | block_32 | 25 / 32 | 78.12% | 78.12% | — | 0.7761 | 2.1729 | — / — |
| baseline_medmcqa | medmcqa | text_oracle | 42 / 42 | complete | block_32 | 32 / 32 | 100.00% | 100.00% | — | 0.0002 | 1.0002 | 100.00% / — |
| baseline_medmcqa | medmcqa | text_tfidf | 42 / 42 | complete | block_32 | 32 / 32 | 100.00% | 100.00% | — | 0.0002 | 1.0002 | 100.00% / — |
| baseline_synthetic | synthetic | frozen | 42 / 42 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 12.4773 | 262309.5499 | — / — |
| baseline_synthetic | synthetic | lora1_r4 | 42 / 42 | complete | block_32 | 1 / 32 | 3.12% | — | 6.25% | 2.7034 | 14.9308 | — / — |
| baseline_synthetic | synthetic | lora4_r1 | 42 / 42 | complete | block_32 | 3 / 32 | 9.38% | — | 0.00% | 3.1701 | 23.8094 | — / — |
| baseline_synthetic | synthetic | lora4_r4 | 42 / 42 | complete | block_32 | 5 / 32 | 15.62% | — | 6.25% | 2.8884 | 17.9645 | — / — |
| baseline_synthetic | synthetic | text_oracle | 42 / 42 | complete | block_32 | 32 / 32 | 100.00% | — | 100.00% | 0.0000 | 1.0000 | 100.00% / — |
| baseline_synthetic | synthetic | text_tfidf | 42 / 42 | complete | block_32 | 32 / 32 | 100.00% | — | 100.00% | 0.0000 | 1.0000 | 100.00% / — |
| vector_medmcqa_transfer | medmcqa | vdb_empty | 42 / 42 | complete | block_32 | 17 / 32 | 53.12% | 53.12% | — | 6.1618 | 474.2658 | 0.00% / — |
| vector_medmcqa_transfer | medmcqa | vdb_oracle | 42 / 42 | complete | block_32 | 17 / 32 | 53.12% | 53.12% | — | 5.7614 | 317.7792 | 100.00% / — |
| vector_medmcqa_transfer | medmcqa | vdb_real | 42 / 42 | complete | block_32 | 17 / 32 | 53.12% | 53.12% | — | 5.7637 | 318.5214 | 3.12% / 6.25% |
| vector_synthetic | synthetic | vdb_empty | 42 / 42 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 12.4773 | 262309.5499 | 0.00% / — |
| vector_synthetic | synthetic | vdb_oracle | 42 / 42 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 2.7557 | 15.7316 | 100.00% / — |
| vector_synthetic | synthetic | vdb_real | 42 / 42 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 2.7604 | 15.8068 | 28.12% / 65.62% |
| vector_synthetic | synthetic | vdb_shuffled | 42 / 42 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 2.7541 | 15.7074 | 28.12% / 65.62% |

Choice accuracy scores the four option candidates and is distinct from generated-output EM. A dash denotes unavailable or invalid evidence, not zero.

## Retention and numerical storage

| Run / method | Priming EM before → after | Control EM before → after | Previous-best minus final EM | Shared trained parameters | Active adapter parameters | Key bytes | Value bytes | Frozen projection bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_medmcqa / frozen | 56.25% → 56.25% | 43.75% → 43.75% | 0.00% | 0 | — | — | — | — |
| baseline_medmcqa / lora1_r4 | 81.25% → 75.00% | 37.50% → 25.00% | 4.17% | 49152 | 49152 | — | — | — |
| baseline_medmcqa / lora4_r1 | 75.00% → 65.62% | 50.00% → 43.75% | -8.33% | 49152 | 12288 | — | — | — |
| baseline_medmcqa / lora4_r4 | 90.62% → 75.00% | 50.00% → 31.25% | 0.00% | 196608 | 49152 | — | — | — |
| baseline_medmcqa / text_oracle | 96.88% → 96.88% | 43.75% → 43.75% | 0.00% | 0 | — | — | — | — |
| baseline_medmcqa / text_tfidf | 96.88% → 96.88% | 50.00% → 43.75% | 0.00% | 0 | — | — | — | — |
| baseline_synthetic / frozen | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| baseline_synthetic / lora1_r4 | 6.25% → 0.00% | 12.50% → 0.00% | 8.33% | 49152 | 49152 | — | — | — |
| baseline_synthetic / lora4_r1 | 0.00% → 6.25% | 0.00% → 12.50% | 8.33% | 49152 | 12288 | — | — | — |
| baseline_synthetic / lora4_r4 | 9.38% → 6.25% | 6.25% → 6.25% | 0.00% | 196608 | 49152 | — | — | — |
| baseline_synthetic / text_oracle | 100.00% → 100.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| baseline_synthetic / text_tfidf | 100.00% → 100.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| vector_medmcqa_transfer / vdb_empty | 56.25% → 56.25% | 43.75% → 43.75% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_medmcqa_transfer / vdb_oracle | 62.50% → 62.50% | 43.75% → 43.75% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_medmcqa_transfer / vdb_real | 62.50% → 62.50% | 50.00% → 50.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_empty | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_oracle | 9.38% → 9.38% | 0.00% → 0.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_real | 9.38% → 9.38% | 6.25% → 6.25% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_shuffled | 9.38% → 9.38% | 6.25% → 6.25% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |

Paraphrase scores were corrected after detecting an evaluation-call bug for: baseline_synthetic/frozen, baseline_synthetic/lora1_r4, baseline_synthetic/lora4_r1, baseline_synthetic/lora4_r4, baseline_synthetic/text_oracle, baseline_synthetic/text_tfidf. These are posthoc implementation-bug corrections evaluated with frozen checkpoints; no retraining was performed. Original mislabeled scores remain identified in the JSON and are not used as paraphrase evidence.

Forgetting is calculated per block as its maximum earlier evaluation EM minus its final EM, then averaged over blocks with an earlier measurement. Negative values indicate improvement. A block first evaluated at the final snapshot is excluded from this average. The JSON retains every block and the alternative nonnegative best-including-final statistic.

## Paired exploratory comparisons

Differences are first minus second. These 10,000 paired fact-bootstrap replicates (bootstrap seed 123) are descriptive intervals conditional on these checkpoints, not uncertainty over model training or independent replications. Pairing requires the same fact IDs, labels, online evaluation seed, final write count and compatible observation hashes. Training seeds may differ and are recorded separately; different online evaluation seeds are never paired.

| First | Second | Paired facts | EM difference | 95% percentile interval |
|---|---|---:|---:|---:|
| vector_synthetic/vdb_real | baseline_synthetic/frozen | 32 | 0.00% | [0.00%, 0.00%] |
| vector_synthetic/vdb_real | baseline_synthetic/lora1_r4 | 32 | -3.12% | [-9.38%, 0.00%] |
| vector_synthetic/vdb_real | baseline_synthetic/lora4_r1 | 32 | -9.38% | [-21.88%, 0.00%] |
| vector_synthetic/vdb_real | baseline_synthetic/lora4_r4 | 32 | -15.62% | [-28.12%, -3.12%] |
| vector_synthetic/vdb_real | baseline_synthetic/text_oracle | 32 | -100.00% | [-100.00%, -100.00%] |
| vector_synthetic/vdb_real | baseline_synthetic/text_tfidf | 32 | -100.00% | [-100.00%, -100.00%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_empty | 32 | 0.00% | [0.00%, 0.00%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_oracle | 32 | 0.00% | [0.00%, 0.00%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_shuffled | 32 | 0.00% | [0.00%, 0.00%] |

## Limitations

- Each method is represented by one training realization: fact-bootstrap intervals condition on the observed checkpoints and online evaluation seed; they do not capture training randomness or support population-level method superiority. Different evaluation seeds are not pooled.
- Synthetic entity-to-value associations share 16 output words and few templates; these results do not establish open-domain memory or held-out-value generalization.
- MedMCQA vector runs transfer an interface trained on synthetic facts; domain-transfer failure alone does not reject the architecture. Public QA may overlap pretraining.
- Answer-token NLL is sum of answer-token negative log likelihood divided by the number of scored answer tokens; PPL is exp of that NLL. Neither is full-prompt or full-corpus perplexity.
- Vector real/shuffled retrieval metrics refer to the final prompt token, not all forward/generation tokens. Oracle selection is a separate diagnostic.
- A rolled value bank changes record identity but can preserve the same answer word; shuffled is not guaranteed to be wrong-valued for every query.
- The unwritten synthetic control labels are random hidden values; their EM measures unexposed guessing rather than abstention accuracy.
- Parameter counts, bank vector bytes and frozen projection bytes are distinct costs; this pilot does not establish matched total storage or deployment latency.
