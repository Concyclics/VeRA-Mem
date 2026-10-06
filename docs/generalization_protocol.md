# Expression generalization: frozen experimental protocol

This round tests whether data augmentation reduces VeRA-Mem's overfitting to one question form. The previous long-training arm reached 127/128 on the original template, but both real and forced-value retrieval scored 0/128 on the old paraphrase. Forced values are not a strict upper bound because forcing the same value at every token changes the real retrieval distribution.

## Architecture and comparisons

Keep the fixed Qwen3-4B-Instruct-2507 version, zero-indexed layer-20 down_proj, rank/key=64, top-k=4, and temperature 0.05. Each actual token's VeRA input generates the query; an observation generates a persistable key/value; sparse values modulate the VeRA branch with frozen random A/B. Shared weights remain frozen online; only the VDB is written.

| Condition | Training question / observation forms | Consistency term |
| --- | --- | --- |
| canonical | 1 / 1 | None |
| augment | 8 / 4, independently sampled | None |
| invariant | Exactly the same primary-view sequence as augment | Query/key cosine and value MSE, each weighted 0.1 |

All three arms use the same 4,096 entities, 400 addressing warm-up updates, 1,536 LM updates, and batch=8 (12,288 target-example exposures). The first 768 LM updates use a correct-value curriculum; the last 768 use real retrieval. All use seed 42. Independent RNGs match target/negative entities across arms and primary views across the augmented arms; auxiliary consistency views are sampled separately. Question lengths and consistency computation differ. This is therefore **matched update and target-example budgeting**, not exactly matched FLOPs. Record prompt-token counts and time.

Every stage retains InfoNCE between different entities. The real-retrieval stage additionally supervises addressing at actual answer-prediction positions. Each episode has one candidate column per entity; second views are not mistakenly treated as negatives. Consistency does not replace LM supervision or negatives. Means are fitted only on offline training views and never adjusted online.

## Held-out sets

Training retains the fixed 4,096-fact pool from seed 1042. The 64-entity development set uses new seed 12042; the 128-entity confirmation set and the 32 never-written controls use seed 17042. Training, development, and confirmation entities are disjoint; the 16 output words are shared. This tests entity and expression generalization, not unseen answer vocabulary, real open-domain knowledge, or long-document capability.

The 8 training question forms include the old paraphrase already inspected in previous work. Development uses card formats and leading output constraints. Confirmation uses XML, CSV, and dialogue families absent from training and checkpoint selection. These are synthetic expression stress tests, not a representation of the entire natural-language distribution. Templates and fact lists are in `augmentation_data.py`; caches record the complete protocol fingerprint.

Select the checkpoint by mean real-retrieval answer NLL over 5 fixed development query/support combinations: original question/original observation, two development questions/original observation, and the two development questions with their corresponding development observations. Test EM does not select the training step.

## Online evaluation and interpretation

Start each run with an independent empty VDB and reveal 128 facts sequentially. One bank uses original observations; another cycles through three confirmation observation forms by fact. The latter is a **mixed-observation-format bank**, not three separate banks each covering all 128 facts. For the canonical-observation bank, record before-write, immediate post-write, and retention performance after all writes, plus before/after behavior on never-written controls. These controls do not measure abstention capability.

For each complete bank, test original questions and three confirmation question forms with real retrieval, forced correct values, shuffled value associations, and zero residual. No retrieved text enters the prompt. A cache may contain future frozen features, but a fact is neither encoded into key/value nor written to the bank before its explicit reveal boundary.

The four main combinations are original/new questions × original/new observations. Report per-template exact match and first-token Recall@1/@4. When aggregating three question forms, average within each entity first. Estimate condition differences with entity-clustered paired bootstrap: 10,000 resamples, seed 123, 95% intervals. Different templates of one fact are not independent samples. Intervals measure only test-fact sampling uncertainty and do not replace replication across training seeds.

A meaningful improvement should jointly show better real-retrieval EM on unseen expressions, retention on the original template, and substantially lower shuffled/empty performance. Report partial improvements explicitly; accuracy on a seen template is not generalization.
