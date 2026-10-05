# Preliminary memory experiment results

Suite status: **complete**.

Only aggregate measurements are published here. A partial suite is not evidence that unfinished methods failed.

## Final observed-stream performance

For a partial method, the final phase is the greatest available block number; it may be earlier than the requested end of the stream.

| Run | Dataset | Method | Training / evaluation seed | State | Phase | Correct / n | EM | Choice accuracy | Paraphrase EM | Answer-token NLL | Answer-token PPL | Retrieval @1 / @k |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_synthetic | synthetic | frozen | 43 / 43 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 12.6386 | 308244.5391 | — / — |
| baseline_synthetic | synthetic | text_oracle | 43 / 43 | complete | block_32 | 32 / 32 | 100.00% | — | 100.00% | 0.0000 | 1.0000 | 100.00% / — |
| baseline_synthetic | synthetic | text_tfidf | 43 / 43 | complete | block_32 | 32 / 32 | 100.00% | — | 100.00% | 0.0000 | 1.0000 | 100.00% / — |
| vector_synthetic | synthetic | vdb_empty | 42 / 43 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 12.6386 | 308244.5391 | 0.00% / — |
| vector_synthetic | synthetic | vdb_oracle | 42 / 43 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 11.2634 | 77917.7453 | 100.00% / — |
| vector_synthetic | synthetic | vdb_real | 42 / 43 | complete | block_32 | 1 / 32 | 3.12% | — | 0.00% | 2.9449 | 19.0084 | 0.00% / 6.25% |
| vector_synthetic | synthetic | vdb_shuffled | 42 / 43 | complete | block_32 | 0 / 32 | 0.00% | — | 0.00% | 15.5020 | 5400543.7113 | 0.00% / 6.25% |

Choice accuracy scores the four option candidates and is distinct from generated-output EM. A dash denotes unavailable or invalid evidence, not zero.

## Retention and numerical storage

| Run / method | Priming EM before → after | Control EM before → after | Previous-best minus final EM | Shared trained parameters | Active adapter parameters | Key bytes | Value bytes | Frozen projection bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_synthetic / frozen | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| baseline_synthetic / text_oracle | 100.00% → 100.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| baseline_synthetic / text_tfidf | 100.00% → 100.00% | 0.00% → 0.00% | 0.00% | 0 | — | — | — | — |
| vector_synthetic / vdb_empty | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_oracle | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_real | 6.25% → 6.25% | 6.25% → 12.50% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |
| vector_synthetic / vdb_shuffled | 0.00% → 0.00% | 0.00% → 0.00% | 0.00% | 1870336 | — | 16384 | 16384 | 3145728 |

Forgetting is calculated per block as its maximum earlier evaluation EM minus its final EM, then averaged over blocks with an earlier measurement. Negative values indicate improvement. A block first evaluated at the final snapshot is excluded from this average. The JSON retains every block and the alternative nonnegative best-including-final statistic.

## Paired exploratory comparisons

Differences are first minus second. These 10,000 paired fact-bootstrap replicates (bootstrap seed 123) are descriptive intervals conditional on these checkpoints, not uncertainty over model training or independent replications. Pairing requires the same fact IDs, labels, online evaluation seed, final write count and compatible observation hashes. Training seeds may differ and are recorded separately; different online evaluation seeds are never paired.

| First | Second | Paired facts | EM difference | 95% percentile interval |
|---|---|---:|---:|---:|
| vector_synthetic/vdb_real | baseline_synthetic/frozen | 32 | 3.12% | [0.00%, 9.38%] |
| vector_synthetic/vdb_real | baseline_synthetic/text_oracle | 32 | -96.88% | [-100.00%, -90.62%] |
| vector_synthetic/vdb_real | baseline_synthetic/text_tfidf | 32 | -96.88% | [-100.00%, -90.62%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_empty | 32 | 3.12% | [0.00%, 9.38%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_oracle | 32 | 3.12% | [0.00%, 9.38%] |
| vector_synthetic/vdb_real | vector_synthetic/vdb_shuffled | 32 | 3.12% | [0.00%, 9.38%] |

## Limitations

- Each method is represented by one training realization: fact-bootstrap intervals condition on the observed checkpoints and online evaluation seed; they do not capture training randomness or support population-level method superiority. Different evaluation seeds are not pooled.
- Synthetic entity-to-value associations share 16 output words and few templates; these results do not establish open-domain memory or held-out-value generalization.
- MedMCQA vector runs transfer an interface trained on synthetic facts; domain-transfer failure alone does not reject the architecture. Public QA may overlap pretraining.
- Answer-token NLL is sum of answer-token negative log likelihood divided by the number of scored answer tokens; PPL is exp of that NLL. Neither is full-prompt or full-corpus perplexity.
- Vector real/shuffled retrieval metrics refer to the final prompt token, not all forward/generation tokens. Oracle selection is a separate diagnostic.
- A rolled value bank changes record identity but can preserve the same answer word; shuffled is not guaranteed to be wrong-valued for every query.
- The unwritten synthetic control labels are random hidden values; their EM measures unexposed guessing rather than abstention accuracy.
- Parameter counts, bank vector bytes and frozen projection bytes are distinct costs; this pilot does not establish matched total storage or deployment latency.
