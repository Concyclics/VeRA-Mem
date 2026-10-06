# VeRA-Mem: Wikipedia cold start and learnable foundation memory

**2026-10-06: all eight main experiments and one supplementary experiment are complete.** Executed under the [fixed protocol](coldstart_protocol.md), formal results comprise 13 actual training jobs and 57 evaluation configurations. Independent audits verified 33,216 generations and 9,960 single-fact updates. The public [key results](results/coldstart/key_facts.json) and [full aggregate summary](results/coldstart/summary.json) support the tables below.

**Conclusion: cold start improved answer-token probabilities, and learnable vectors/dense backward passes expanded gradient coverage, but this round did not improve complete factual answers or format generalization.** On teacher-qualified synthetic confirmation tasks, all nine models score **0/64** in every one of the four expression conditions. Small probes of previously trained facts also score **0/8** throughout. The current degeneration therefore cannot be explained solely by foundation vectors being too sparsely updated. Information preservation during writing, readout capacity, and decode-time routing still require separate tests; no root cause is established here.

This is a supervised cold-start experiment with **32K candidates, 8,192 warm-up targets, one seed, and one rank-64 layer**, not a conclusion about Wikipedia-scale pretraining. The original Wikipedia-context teacher is itself unqualified. Additional answer annotation raises training-validation teacher qualification to 93.75%, but its supplementary student still gains no complete answers. Wiki zero scores alone cannot isolate student generalization failure or rule out larger-scale pretraining or dynamic clustering during training.

## Architecture and the three proposed changes

Inference preserves the requested interface: each token's actual VeRA-layer input produces a query; sparse CPU-VDB retrieval returns values that parameterize VeRA modulation. A shared writer encodes new observations into keys/values that can be written and replaced by record. This path does not append retrieved text to the student's prompt.

Two memories are separated. The **dynamic factual bank** stores independently replaceable new observations. The **foundation bank** stores shared parameter slots initialized or trained offline. In foundation-enabled arms, each bank retrieves top-4; `v_dynamic + 0.25 × v_base` enters the same VeRA gate, accessing at most eight records. The first three controls set the foundation coefficient to zero. Disabling dynamic memory does not disable foundation memory, and a zero foundation coefficient does not automatically eliminate the implementation's foundation-query cost. Online evaluation freezes shared weights and changes only designated factual VDB records.

| Proposed change | Actual controls | What they test |
| --- | --- | --- |
| Initialize VeRA and foundation memory from Wikipedia | `no_warm`, `matched_warm`, `wiki_warm`; initialize foundation slots by clustering Wiki A observations; additional `wiki_joint` | No gradient warm-up versus compact observations with the same targets versus natural paragraphs; whether foundation slots participate in corpus warm-up |
| Learn foundation vectors to address inadequate sparse-gradient coverage | `fixed`, `learned_sparse`, `dense_warm`, `straight_through` | Frozen slots, ordinary sparse training, dense-to-sparse training, and sparse forward/dense backward from common initialization |
| Cluster or merge vectors | `full1024`, `off`, `random256`, `cluster256` on one prespecified ST checkpoint | Capacity reduction and merging effects, without equating higher coverage in a smaller bank with better quality |

Output-level straight-through uses `y_dense + stop_gradient(y_sparse − y_dense)`: the forward pass remains sparse, while backward gives gradients to unselected values too. This is a biased surrogate-gradient diagnostic with dense computational cost. Clustering compresses only the foundation bank and retains arithmetic value means, without extra RMS rescaling or count bias. Conflicting new facts are never averaged together.

Clustering is used only for pretraining prototype initialization and post-training compression of a fixed checkpoint. This round does not test dynamic merging, splitting, or reallocation of inactive slots during training, so it cannot rule out those anti-degeneration mechanisms.

## Eight main arms and one supplementary arm

| Main arm | Corpus warm-up | Subsequent synthetic training |
| --- | --- | --- |
| `no_warm` | No gradient warm-up; still shares statistics fitted on training-corpus features | 512 steps, foundation contribution disabled |
| `matched_warm` | 1,024 steps of compact templated observations, sharing Wiki IDs, questions, A/B answers, and target order | 512 steps, foundation contribution disabled |
| `wiki_warm` | 1,024 steps of natural-paragraph observations | 512 steps, foundation contribution disabled |
| `fixed` | Reuses common Wiki warm-up and its 1,024 prototypes | 512 steps, prototypes frozen |
| `learned_sparse` | Same as `fixed` | 512 steps, trainable prototypes and ordinary sparse routing |
| `dense_warm` | Same as `fixed` | 256 dense steps, then 256 sparse steps |
| `straight_through` | Same as `fixed` | 512 output-level ST steps |
| `wiki_joint` | Builds prototypes from the common initial model; jointly trains prototypes and VeRA for 1,024 Wiki steps using ST | Continues ST for 512 steps |

`wiki_joint` changes both foundation-memory use and routing during warm-up, so it is not a single-factor control. The eight arms require **11 training jobs and 52 evaluation configurations**: 22 development, 22 confirmation, and 8 training-memory probes. Shared warm-up is counted only once in actual compute cost.

The supplementary **`wiki_teacher_repair`** arm starts after teacher failure under a prespecified training-set rule, independently of the eight main arms: common initialization → 1,024 Wiki warm-up steps with an answer-annotated teacher → initialization of 1,024 prototypes → 512 synthetic ST steps, restoring the ordinary teacher for synthetic training. It adds two training jobs, one prototype initialization, and five evaluation configurations. Student observations, feature caches, questions, target order, and the inference interface remain unchanged; Wiki teacher supervision changes. It must not be pooled with the original natural-context controls to claim that simply adding corpus data helped.

All branches use frozen Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, layer 20, fixed random A/B, rank 64, and seed 63042. Writer values use pooling over content spans; keys use the observation's last position. Training uses only the canonical view, with P=A, so this is not independent paraphrase augmentation.

Shared objectives include forward KL from a contextual teacher, full gold-answer CE weighted by 0.5, and target-address supervision. Original Wiki warm-up is therefore supervised training, not unsupervised language pretraining. Natural paragraphs add content diversity, not additional training-question or observation-format types.

## Data scale, isolation, and known limitations

Wiki uses a fixed Wikimedia `20231101.en` shard. From 156,289 source articles, it yields **32,768/64/128** natural observations for train/dev/confirm; matched templates preserve labels and questions. A is the original article excerpt; B counterfactually replaces its three-word answer with another article's answer from the same split. This tests factual updates conditioned on observations, not ordinary encyclopedic QA.

**A 32K corpus does not mean 32K supervised targets.** Wiki/template warm-up uses 1,024 × 8 = **8,192 distinct targets, each updated once**, a subset of 32,768 candidates. Foundation-prototype initialization uses all 32,768 A observations. Common normalization statistics use only the first 8,192 rows. Synthetic training uses 512 × 8 = **4,096 distinct targets, each updated once**, exactly one traversal; its train/dev/confirm sizes are 4,096/64/128. Background-bank exposure and A/B/P branches of the same target are not additional independent epochs. Exact input/target token totals appear in the cost table; record counts are not token counts.

Article IDs and exact duplicate clusters based on normalized titles/bodies/lead paragraphs are split first; complete answers are disjoint across splits. Synthetic entities and complete answers are also isolated. This is not exhaustive semantic near-duplicate exclusion, and Qwen may have encountered the original encyclopedia text in pretraining. Development evaluates 16 targets per domain; confirmation fixes 64 targets from each 128-record bank. Training-memory probes evaluate 8 of 128 previously trained facts, copying the canonical view for interface compatibility. Their four phases therefore use the same format and do not measure format generalization. Probe banks of 128 background facts are not exact replays of 72-record training episodes. Changed negatives can also alter routing, so probe failure is not identical to proving that no training example was memorized.

Data validation found and addressed two issues:

- At Wiki train index 13,896, the B replacement overlapped the anchor; that record became a warm-up target at step 940. Both affected old Wiki branches were excluded in full and preserved. The v2 repair changes only this B record's two raw text views and canonical-B last/pool features. Queries, all A features, all other features, and shared-statistics inputs remain bitwise unchanged; the other eight raw data files are byte-identical. The matched-template branch was unaffected.
- The A answer `David Croft and` at train index 6,295 occurs inside the punctuation-normalized question anchor. Wiki and matched data share this limitation. Main data were not deleted or changed post hoc. The record was not among this round's 8,192 training targets, but remains in shared-statistics/prototype source data. Whole-phrase scans found no such issue in dev/confirm or synthetic train. A fixed rule excluded it from teacher probing before predictions, leaving 32,767 eligible candidates.

## Teacher feasibility must precede interpretation of student outcomes

| Completed teacher check | baseline | Extraction instruction `quote_instruction` | Answer annotation `gold_annotated` |
| --- | --- | --- | --- |
| 16 Wiki training calibration records, both A/B correct | 0/16 | 0/16 | 16/16 |
| Separate 64 Wiki training-validation records, both A/B correct | 1/64 | 0/64 | **60/64 (93.75%)** |

The selection rule was sealed before probe predictions: use quote if it reaches 90%; otherwise use annotation only if it reaches 90%; otherwise add no training. A completed independent audit recomputed paired counts from **480 raw generations**, checked fixed candidate filtering, unchanged questions/source contexts, and evidence/source-snapshot hashes. Selecting `gold_annotated` complied with the rule. Annotation explicitly supplies answer localization and is **additional label supervision**, not an unannotated Wiki-context teacher or unsupervised corpus learning.

The original Wiki development teacher scores only **0/16, 0/16, 1/16, 0/16** for CC/CH/HC/HH; the synthetic teacher scores **16/16** in all four phases. Under the current prompt and generation budget, the original Wiki teacher is unqualified, so its condition cannot independently isolate student memory-generalization failure. This does not make the task unlearnable or imply that gold-prefix teacher KL is wholly useless; full-answer CE and address supervision are still present. A 93.75% annotated-teacher score on the training distribution neither qualifies it on new articles nor establishes that the parameterized student learned to read/write.

## Final confirmation: probability gains have not become complete answers

The first CC/CH/HC/HH letter denotes observation format and the second question format. C is the canonical training format; H is a held-out format. Paired EM requires complete correct answers to the same question after both A writing and B replacement. NLL below is answer-token-weighted within each world and then averaged equally across A/B; lower is better. It is scored under correct-answer prefixes and does not replace free generation.

| Model | Synthetic confirmation paired EM (CC/CH/HC/HH) | Synthetic NLL: CC | HC | HH |
| --- | --- | --- | --- | --- |
| `no_warm` | 0/64 × 4 | 4.415 | 4.593 | 8.430 |
| `matched_warm` | 0/64 × 4 | 3.820 | 3.898 | 5.241 |
| `wiki_warm` | 0/64 × 4 | 4.083 | 4.257 | 5.524 |
| `fixed` | 0/64 × 4 | 3.911 | 4.060 | 5.778 |
| `learned_sparse` | 0/64 × 4 | 3.862 | 3.988 | 5.864 |
| `dense_warm` | 0/64 × 4 | 3.892 | 4.012 | 5.737 |
| `straight_through` | 0/64 × 4 | 3.856 | 4.005 | 5.628 |
| `wiki_joint` | 0/64 × 4 | 4.048 | 4.084 | 5.426 |
| Supplementary: `wiki_teacher_repair` | 0/64 × 4 | 3.712 | 3.825 | 5.345 |

Wiki confirmation likewise gives **A=0/64, B=0/64, paired=0/64** for all nine models in all four phases. Its original-context teacher scores **4/64, 0/64, 4/64, 0/64**, whereas the synthetic teacher scores **64/64** throughout. The teacher-qualified synthetic task therefore provides more interpretable failure evidence than Wiki alone.

All four real-bank phases score 0/16 in the 22 main and 2 supplementary development configurations. All nine training-memory probes score 0/8 individually for A/B and jointly, with teachers at 8/8. These probe phases share a format and use a different background bank from training episodes; they are neither four independent generalization tests nor a full training-set fit evaluation.

Real-bank evaluation covers four phases; value permutation and an empty dynamic bank are tested only in CC/HC, and `canonical_key` only in HC. None of these executed controls produces a complete correct answer. `canonical_key` is an auxiliary interface diagnostic, not a deployment method or mathematical upper bound. The prespecified threshold requires HC to improve by at least 10 percentage points with paired EM ≥20%, while CC drops by at most 5 points. **No method passes.**

There are **15,936 formal real-bank outputs**, all with zero full EM and zero full-answer containment. They repeat facts across models, worlds, and expression conditions; they are not 15,936 independent facts. Of 4,096 synthetic-confirmation outputs from the eight main full-bank arms, 3,448 (84.18%) already contain three normalized words and none hits the generation budget. Uniform truncation therefore cannot explain that condition's zero score. Main full-bank Wiki confirmation has 163/4,096 budget hits, whose influence must remain acknowledged. Full length, first-word, and truncation diagnostics are in the [answer diagnosis](results/coldstart/answer_diagnostics.json).

Addressing remains format-sensitive. Across nine models, synthetic-confirmation R@4 at the first answer position, averaged across A/B, is **81.25–86.72%** for CC, **10.94–14.06%** for HC, and **6.25–7.81%** for HH. For example, `learned_sparse` reaches 86.72% in CC but still produces no complete answers. Decode correct-record residency is only about 12.58% (token-weighted separately in A/B, then averaged equally). Residency is total hits divided by total decode queries, giving longer wrong responses more weight. It is neither entity-averaged accuracy nor direct proof that routing drift causes failure.

## Foundation vectors update, but coverage is not effective utilization

The vector audit covers all 13 formal training jobs, with smoke tests separate. `fixed` foundation keys/values remain identical to its parent checkpoint. `fixed/learned_sparse/dense_warm/straight_through` share the same initial foundation-bank hash. The table describes the subsequent 512 synthetic steps, with the supplement separate:

| Arm | Union of key/value slots with nonzero gradients | Slots selected as top-1 | Entropy-effective slots from top-1 counts | Entropy-effective slots from value Adam RMS mass |
| --- | --- | --- | --- | --- |
| fixed | 0 / 0 | 785/1024 | 95.6 | Not applicable |
| learned_sparse | 1001 / 1001 | 422/1024 | 51.9 | 131.6 |
| dense_warm | 1024 / 1024 | 486/1024 | 40.6 | 139.6 |
| straight_through | 1024 / 1024 | 436/1024 | 41.0 | 119.9 |
| wiki_joint | 1024 / 1024 | 105/1024 | 1.4 | 15.6 |
| Supplementary wiki_teacher_repair | 1024 / 1024 | 326/1024 | 28.8 | 93.6 |

ST receives more than tiny nonzero gradients: median relative foundation-value update norm is approximately **8.45%**, versus **7.58%** in the supplement, still without complete answers. In `wiki_joint`, every slot updates, with a median around 33.0%, but top-1 use and final Adam mass are strongly concentrated. The first singular direction of its uncentered value-update matrix accounts for 89.57% of squared energy (31.13% after centering). Yet the effective rank of final centered values changes from about 22.43 to 22.90. **This is not collapse of the entire vector bank to rank one.**

Routing counts use only the last student forward of each update and may include padding; they are not counts over all training tokens or first-answer tokens. The Adam column computes per-slot `m_i = sqrt(mean_j(exp_avg_sq[i,j]))`, normalizes it, then takes `exp(entropy)`. It uses the final exponentially averaged second moment of clipped gradients, not raw gradients or full-trajectory gradient energy. These metrics show changes in optimization/utilization distributions, not semantic capacity or resistance to degeneration. See the [vector audit](results/coldstart/vector_audit.json) for initial/final tensors, gradients, and effective-rank evidence.

## Clustering compression: no generation gain yet

One prespecified ST checkpoint gives the following confirmation results:

| Foundation-bank mode | Slots | Synthetic CC NLL | Wiki CC NLL | Paired EM across both domains and four phases |
| --- | --- | --- | --- | --- |
| full | 1024 | 3.856 | 5.986 | All 0/64 |
| off | 0 contribute to mixing | 4.245 | 6.210 | All 0/64 |
| random | 256 | 3.915 | 6.019 | All 0/64 |
| cluster | 256 | 3.886 | 5.991 | All 0/64 |

Turning foundation contribution off worsens NLL, establishing an effect on this probability metric. Cluster compression stays closer to the full bank than the random subset, but all four fail to produce complete answers. This does not establish preserved capability or improved generalization. Clustering gives equal weight to each learned prototype, not its original Wiki cluster population. Identical first-position episodic R@4 follows structurally because this query is computed before memory injection at the layer; it does not show that compression leaves later generation unchanged.

## Costs, exclusions, and reproducibility

| Scope | Updates/generations | Recorded tokens / processed positions | Process seconds |
| --- | --- | --- | --- |
| 11 main training jobs, shared prefixes counted once | 7,168 updates, 57,344 target exposures | 44,252,178 input positions; 43,008 backbone calls | 2,651.09 |
| 52 main evaluation configurations | 30,528 free generations; 9,120 single-record B updates | 227,601 generated tokens; 139,713 answer-scoring target tokens | 10,406.92 (generation only) |
| 2 supplementary training jobs | 1,536 updates, 12,288 target exposures | 13,332,908 input positions; 9,216 backbone calls | 730.44 |
| 5 supplementary evaluation configurations | 2,688 free generations; 840 single-record B updates | 16,298 generated tokens; 12,366 answer-scoring target tokens | 516.97 (generation only) |
| Independent teacher-feasibility probe | 480 generations and 480 scoring calls | 4,445 generated tokens; 1,758 scoring-target tokens; 329,681 input positions | 134.50 generation + 19.41 scoring |
| Excluded old Wiki v1 branches | 3,072 updates, 24,576 target exposures | 23,683,104 input positions; 18,432 backbone calls | 1,353.62 |

Training inputs sum processed positions from student, teacher, and rollout calls, including repeated processing. Gold, distillation, and target tokens overlap and must not be added again. Training elapsed time uses the final cumulative value. Summed concurrent process seconds are not wall time or exclusive GPU latency, and generation seconds exclude answer scoring and other overhead. Feature extraction, model loading, prototype initialization, repair, transfer, and CPU audits lack a unified end-to-end cost aggregation; consult their manifests. This table is not total project cost or evidence of efficiency superiority. The answer-annotated teacher also increases warm-up input positions, so it is not exactly compute-matched to the original Wiki teacher.

Smoke tests add 4 training updates, 48 generations, and 10 single-record updates, excluded from formal effects and the main/supplementary totals above. Excluded data-v1 runs, the first failed teacher precheck, one failed resource precheck, and two old-CLI launch rejections remain preserved and are not counted as zero-scoring experiments. The full summary includes smoke tests. Read `formal_training_costs` for formal training and `supplementary_teacher_followup` for the supplement rather than treating overall totals as costs of the eight main arms.

All final independent checks pass:

- [Data, budget, and lineage audit](results/coldstart/independent_audit.json): complete coverage of main 11 training/52 evaluation jobs and supplementary 2 training/1 initialization/5 evaluation jobs; v2 repair, splits, target schedules, individual generation metrics, and frozen-parameter checks pass.
- [VDB replay](results/coldstart/bank_replay.json): 58 evaluation configurations, 33,264 generations, and 9,970 single-record updates pass, including separately labeled smoke tests. Formal counts are 33,216/9,960. Checks cover target-row-only A→B updates, zero teacher VDB queries, shared dynamic banks across compression conditions, and random-subset mappings. Maximum recomputed cluster-value difference is 8.86e-7. This audit reconstructs bank tensors and log contracts; **it does not rerun language-model generation or actual query logits**.
- [Supplementary-teacher lineage](results/coldstart/teacher_lineage.json): selection evidence sealed before training, with stage-by-stage verification from common initialization through warm-up, prototypes, and transfer checkpoints. Student questions, observation features, and targets are unchanged.
- Code validation: the full pytest suite passed **764 tests**. After adding a formal-cost-accounting test, **24 related tests passed**, and final collection contained 765. Vector, bank-replay, answer-diagnosis, and scheduling-script self-tests pass. Passing tests are not evidence of model capability.

[Executed plans and teacher-selection evidence](results/coldstart/plans.json) retain historical input paths and original plan hashes. Reproduction requires new output directories and fresh resource checks. Public audit JSONs are aggregate projections retaining full-original SHA-256 values. Complete per-slot arrays, raw predictions, lightweight checkpoints, source snapshots, and data stay in the Workspace. Full summary-original SHA-256: `d1a75722a447acbc22853c486700310e0513a9fe0ce7cbf1915a4d5b88fe246c`. Remote artifacts are at `xtrah100:/ssd3/chenhan/VeRA-Mem-Workspace`; local artifacts are at `/mnt/storage/chenhan/xtraNet/VeRA-Mem-Workspace`. Backups exclude open-source base weights and environment caches. The Workspace's `backup_manifest.json` is authoritative for final synchronization status and checksum differences across the four directory categories.

## Next steps for students: validate writing/readout before expanding scale

The following are next-round recommendations, **not implemented or validated in this round**. Retain actual-layer-input → query → sparse VDB → value parameters → VeRA and independent fact overwrites.

1. **Reconstruct observed content before new-fact QA.** On a fixed small training set, require the written parameter memory to reconstruct or complete observed content and establish learning of short text and A/B updates. Only then expand the corpus and test new articles. Make this a separate training stage/feasibility diagnostic and report actual generation, rather than stacking further auxiliary losses onto the failed configuration. [SHINE](https://arxiv.org/abs/2602.06358) is a nearby content-to-parameters reference, but its hypernetwork/LoRA path is not direct evidence for this sparse VDB.
2. **Test whether one pooled value loses multiword information.** Change only the writer: compare one vector per observation with a small number of content memory tokens/vector groups. Atomically replace the whole group under one fact ID, preserving A/B conflict controls. Report write bytes, retrieval counts, and added compute with budget-matched controls. Keep actual per-token routing in complete generation. Fixed-prefix/fixed-route auxiliary controls may diagnose writing losses but cannot substitute for deployment results.
3. **Test readout capacity separately.** Once teachers qualify and seen facts can be fitted, fix the writer/data budget and compare single-layer fixed B, learned shared B, and multilayer readout at approximately matched total rank. Change factors individually and track seen-fact fitting, format generalization, and old-fact retention; do not assume fixed B is an established cause. [Memory Layers at Scale](https://arxiv.org/abs/2412.09764) motivates multilayer memory, and [pretrained PKM](https://aclanthology.org/2020.findings-emnlp.362/) motivates initialization/utilization diagnostics. Neither establishes one-shot write generalization for this design.

This round uses one optimization seed, one frozen backbone, one fixed random rank-64 B, and a limited budget. Online confirmation banks contain 128 facts plus 1,024 foundation slots; large ANN indexes and long continuous write streams are not validated. At a frozen checkpoint, this layer's memory residual lies in the at-most-64-dimensional column space of `diag(b)B`; adding foundation slots alone does not expand it. Because b is learned during training, that subspace must not be described as permanently fixed throughout training. This architectural fact motivates a hypothesis; **it is not a causal explanation established by the failures**. Establish reproducible writing/readout of seen facts before investing in more corpus data, foundation slots, or training. See the [literature map](coldstart_literature.md) for further evidence and applicability boundaries.
