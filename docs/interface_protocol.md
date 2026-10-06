# Vector-memory interfaces and training experiments

This round follows four steps: four-bank interventions, writer changes, factual addressing with multi-token answers, and distillation ablations. The aim is to locate cross-expression failures and test limited changes, rather than repeatedly selecting recipes on confirmation scores. The backbone is frozen Qwen3-4B-Instruct-2507 at the historical fixed revision; layer-20 down_proj, rank/key 64, and top-k 4 remain unchanged. Runs execute on H100 SSD3, preserving source hashes, actual commands, and lightweight checkpoints, with local backups.

## Four-bank interventions

Use 64 new development facts from seed 47042. Read original questions and keep all other records canonical. The target record uses KcVc, KhVc, KcVh, or KhVh, with JSON and Markdown as the two new support structures. Each A/B world independently modifies one target record. For value-only comparisons under the same question and keys, check that actual first-position top-k indices and weights are bitwise identical; report free generation separately. The original-text teacher has a 16-token budget. This is a development diagnostic, not the subsequent confirmation set.

<a id="writer-validation"></a>
## Writer validation

The independent 4096-record training split retains historical facts. New development entities use seed 48042 and historical natural-language development templates. New confirmation entities use seed 58042, with 128 records and JSON/Markdown supports. Structured queries end with an explicit single-word output instruction; the generation budget is 32. Regenerate features for every writer using the same Remember prefix. Pool support-body tokens only, excluding instructions, system text, chat wrappers, and padding.

Initialize from the existing base checkpoint and freeze readout and addressing parameters. Compare last-linear, mean-linear, and last-MLP256; train only Wv and the optional MLP. Targets are values produced by the frozen old writer from the same fact/world's canonical expression, with independent A/B targets. Each arm receives 2000 updates with batch size 128, Adam at 1e-4, gradient clipping at 1, and training seeds 42/43/44, using the final step. Mean pooling separately fits a training-value center. Record actual trainable parameter counts; this is not an equal-parameter-capacity comparison.

Development queries use only 32 fixed targets while retaining a 128-record bank. Rank writers first by real HC paired accuracy, then canonical-key diagnostic HC paired accuracy, then CC paired accuracy. If still tied, prefer the simpler last-linear writer. Lock selection before confirmation and preserve every development result. The minimum threshold for continuing the main approach is a 20-percentage-point gain over the original model for new supports with original queries, with no more than a 5-point original-format decline, reproduced in all three seeds. This is a project decision threshold.

## Addressing and expanded answers

Use independent entities, four relations per entity, and three-word answers, with complete answers disjoint across train/dev/confirm. Split seeds are 41042/42042/43042. Actual A banks contain hard negatives from other entities matching both target A and target B answers, plus other-relation negatives for the same entity. Training uses only natural-language templates; confirmation includes JSON, Markdown, YAML, and INI. Query/support structures are independently balanced within relations to avoid binding structure to relation identity.

Compare common paired supervision with an additional 0.05 A/B key-consistency term for the same address, using identical data, initialization, and sampling schedules. Each arm receives 512 updates with batch size 8 with three training seeds. Shared parameters are trained offline and frozen for all evaluation. Report this task separately from the historical 16-word task; difficulty changes cannot identify the benefit of one training adjustment. Final confirmation has 256 facts over 64 entities. The four relations of one entity are not four independent entities.

## Distillation ablations

Continue from the expanded-task address-consistency arm's common checkpoint and Adam state. Compare base, the original clip 10 difference, a normalized difference, normalized+hidden, and normalized+25% student trajectories. Calibrate teacher scale only on the first 256 training facts, fixing it to the median absolute eligible teacher difference. The normalized loss is scale times SmoothL1 and preserves variation in teacher magnitudes; it does not divide each example by its own difference.

All five arms share FKL, full-gold-sequence CE, gold-position addressing, cross-expression addressing, and A/P consistency. On-policy changes only the prefix source for FKL; other supervision always uses the common gold branch. Every 4 steps, sample a common prefix from an equal A/B mixture of student policies, with at most 8 tokens. Other branches do not add artificial EOS. This is world-mixture FKL, not a reproduction of the original OPCD paper. Compare hidden and on-policy separately against the normalized arm rather than adding both factors together. There is no periodic replay. Record all additional forwards and tokens; do not claim matched FLOPs.

If the earlier deployment path passes the continuation threshold, run all five arms for 512 updates with three seeds. Otherwise, limit the comparison to seed 42 and 256 updates per arm as a diagnostic budget, explicitly insufficient to establish stable gains. Generate new confirmation answers only after all training finishes, without checkpoint selection on confirmation results.

## Acceptance and interpretation

The primary metric requires strictly correct complete answers in both A/B worlds of the same fact. Separately report one-sided accuracy, first-position R1/R4, decode residency, format/budget failures, and both teacher-qualified and full denominators. Real, shuffle, and empty controls cover CC/HC; describe the other two query-change conditions according to the actual evaluation matrix. Canonical-key is a mechanism diagnostic, not a deployment result. Updating one online record must not alter other records' bytes or timestamps.

Use entity-clustered paired statistics, with the shared construction groups of expanded answers as supplementary clusters. Report all three seeds; sample bootstrap alone is not training uncertainty. State the measured scope of write/read and training costs, without extrapolating small-bank exact search into production-scale VDB advantages. If the threshold fails, retain negative results and stop adding losses to the current structure.

## Execution notes

Before formal expanded-task training, train-only calibration gives scale=71.015625; all 256 eligible teacher differences saturate the old clip 10 threshold. Teacher-interface preflight uses 16 training facts, 20 expression conditions, and A/B worlds, producing 640 generations, all strictly correct. These are not 640 independent facts. The old clipped objective retains each example's A/B token labels and is not devoid of content supervision. The new version both restores unclipped targets and changes the Huber transition scale; the comparison identifies this combined correction, not magnitude diversity alone.

Development only locks the writer; the continuation threshold is evaluated seed by seed on new classic confirmation data. The three training seeds share the historical base initialization, measuring uncertainty from this round's optimization sampling schedules, not backbone, random-projection initialization, or data-split uncertainty. Expanded-task complete answers are disjoint across splits, but their component vocabulary overlaps. This is neither an open-vocabulary nor an arbitrary-document memory experiment.

Confirmation evaluation may run two independent processes on the same available GPU, with separate models and banks and unchanged counts/budgets. Check for other compute processes before launch; when filling available slots, accept only still-live children of this runner. On failure, stop dispatching new work and let already-running evaluations finish naturally. Resource contention affects per-task wall time under parallelism, so it is not a single-request serving-latency comparison.

## Completion record

All four steps completed on 2026-10-06, comprising 22 suites and 53 jobs. The formal writer threshold failed, so step four used the predetermined diagnostic budget of seed 42 and 256 steps per arm. Results and verification limits are in [four-step results and student handoff](interface_results.md). This section records execution status only; it does not change the selection, budget, or evaluation rules above.
