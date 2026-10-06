# Dynamic Block Memory with Low-Rank Operators: Complete Results

**The dynamic block operator produced limited, observable improvements, but did not solve generalization to new content combinations and expressions.** All 96 formal jobs completed:18 models,72 student evaluations,5 teacher jobs, and 1 feature-preparation job. The teacher was correct on 448/448 cases. Every model used its fixed endpoint, and every condition is reported, without selecting arms or seeds after seeing results.

With random rebinding, `block_outer` correctly updated and restored `3/5/2` novel C combinations, versus `0/0/0` for parameter-matched `pooled_outer`. On independent confirmation D, update correctness was `6/7/3` and update-and-restore correctness `6/6/3`, versus `1/0/0` for pooled on both metrics. Constructing operators per slot therefore improved on pooling first for these two sets of novel combinations. However, C did not improve over the original diagonal reader in every seed, new entities and question expressions remained weak, and the preregistered criterion requiring consistency across three seeds was not met. **These results do not justify expanding to Wikipedia.**

Every slash-separated triple in this report lists counts for seeds 91042, 91043, 91044, each with denominator 16; it is not a single ratio. The three seeds share a tiny dataset, and repeated facts are not independent corpus samples. Evidence is in the [complete summary](results/block/summary.json), [compact metrics](results/block/key_facts.json), and [word-level diagnostics](results/block/answer_diagnostics.json). Conditions are recorded in the [sealed protocol](block_protocol.md), [registration](results/block/registration.json), and [complete plans](results/block/plans.json).

> This English localization preserves the historical experiment and its qualifications. The translated protocol does not replace the registered source or hashes; see [reproducibility.md](reproducibility.md).

## 1. What changed in this round

The backbone is Qwen3-4B-Instruct-2507, with every original model parameter frozen and a rank 64 interface at the layer 20 down projection. Actual layer inputs generate queries. The real CPU VDB selects the top-1 fact using QK group scores and returns its complete three 64-dimensional values. All three readers use uniform within-block weights and retain the same diagonal path. CPU retrieval passes all three selected value rows into the reader, which constructs its operator.

Let `z=Ax`, `m=mean_s(v_s)`, `u(v)=L2Normalize(P_in v+c)`, and `w(v)=P_out v`:

| Reader | Latent operator M acting on z |
| --- | --- |
| diagonal | `diag(m)` |
| pooled_outer | `diag(m)+w(m)u(m)ᵀ` |
| block_outer | `diag(m)+mean_s[w(v_s)u(v_s)ᵀ]` |

Every final increment is `Δh=b⊙B[Mz]`. The added outer products allow nondiagonal mixing of latent coordinates; this is not moving shared B before a weighted sum. Without an intervening nonlinearity, that relocation is exactly linearly equivalent and cannot add capacity. See [block_literature.md](block_literature.md) for the literature mapping and [block_protocol.md](block_protocol.md) for fixed formulas, initialization, and reductions.

Both outer arms add 8,256 parameters, totaling 2,042,624 trainable parameters, versus 2,034,368 for diagonal. Pure FP32 K/V storage is 1,536 bytes per fact, or 24 KiB for 16 facts. No additional 64×64 matrix is stored; shared projections generate factors during reading. The added block term has rank≤3 and the pooled term rank≤1, but **the full diagonal-plus-outer operator can still have rank 64**. All three readers ultimately pass through B with shared width 64. Equal parameters and memory bytes do not imply equal FLOPs.

After fact selection, this reader is invariant to permutations of V's rows. It is neither a word-by-word decoder with explicit slot order nor native multihead Transformer KV attention. Contextual word-span vectors can carry original-prefix and order information, so row-permutation invariance does not imply a complete absence of order information. This round did not test multihead readers, unshared slot mappings, multiple native attention layers, or wider B.

## 2. Data, budget, and sealing boundaries

The new data seed is 221042, excluding 208 historical entities and 608 complete answers from reconstruction/QKV. Training uses 64 entities and 128 three-word payloads. Actual entity+payload support texts are re-encoded, yielding 8,192 frozen feature combinations. The writer uses annotated/locatable three-word spans in observed support text. This remains a structured task, not automatic discovery of fact boundaries in arbitrary Wikipedia documents.

Static fixes A/B per entity; rebind reshuffles bindings each epoch. Each model receives 2,048 updates with batch 8 targets, totaling 16 A/B sequences. Each sample has an independent 16-fact bank; B updates only its own target. Every 8 steps cover all 64 entities and 128 payloads. Training uses only full-answer CE including EOS and 0.2 dense address CE. There are 6 arms×3 seeds, or 18 models, all evaluated at their fixed final checkpoint, with no C-based model selection or early stopping.

Known/dev/confirm banks all contain 16 facts; the latter two use new entities. Payloads are shared row by row across the three packets to isolate entity changes. C/D each change one word of A; all component words have been trained, but complete combinations have not. Held-out templates are unseen only within this round's training and have been used in earlier research. D/confirm features may be encoded in advance; their scores do not affect gradients, statistics, or model selection.

Known A/B should be described as “fixed-pair tests of training-seen entities and complete payloads.” Under rebind, these specific bindings received mean target-supervision counts 4.0625,4.40625,4.21875. Seed 91042 has two A bindings never supervised as targets; the other seeds have no such zero entries. Each static binding receives 256 exposures. Zero target exposure does not mean absence from feature encoding, training statistics, or background residence, and should not be called “entirely unseen.”

The teacher is used only for qualification: the original backbone receives the target's single context and the same question, without VDB access, additional gold labels, or distillation training. Final teacher correctness is 448/448:128 training A/B preflight,48 each for known/dev development,32 for known D confirmation, and 192 for new-entity A/B/D across four phases. A/B/D each score 16/16 in every new-entity phase. Thus teacher-qualified subsets equal the full samples for corresponding non-SWAP student conditions; no items were selected by teacher success.

## 3. Development endpoints: partial combination signal, no consistent winner

All entries are correct counts for the three seeds, each out of 16. C `update_restore` requires correct output after writing C and after restoring A. C updated and update_restore happen to match entry by entry in this table; they are not equivalent by definition.

| Arm | Known A/B pair | Known C updated | Known C update_restore | New-entity dev C updated |
| --- | --- | --- | --- | --- |
| static_diagonal | 16/16/16 | 0/2/0 | 0/2/0 | 0/0/0 |
| static_pooled_outer | 16/16/16 | 0/1/0 | 0/1/0 | 0/0/0 |
| static_block_outer | 15/16/16 | 0/0/0 | 0/0/0 | 0/1/1 |
| rebind_diagonal | 15/16/14 | 2/0/4 | 2/0/4 | 0/1/1 |
| rebind_pooled_outer | 15/16/13 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_block_outer | 15/15/14 | **3/5/2** | **3/5/2** | **1/5/0** |

All known C shuffle/empty scores are 0. Rebind block's joint neighbor correctness before/after is 15/15/15. Static arms score 16/16/16; rebind diagonal and pooled both score 15/16/14. This joint metric is stricter than “unchanged output text”: two wrong answers do not count as successful locality.

The following diagnostics require exactly three normalized words. “Both retained words” requires both unchanged words to be correct, without requiring a correct edited word. Successful cases in the two columns can differ and cannot be added.

| Arm | Known C edited word correct | Known C both retained words correct | Dev C edited word correct | Dev C both retained words correct |
| --- | --- | --- | --- | --- |
| static_diagonal | 3/5/3 | 6/5/3 | 2/5/4 | 0/0/0 |
| static_pooled_outer | 2/2/1 | 3/6/6 | 2/2/4 | 1/0/0 |
| static_block_outer | 1/4/5 | 6/6/7 | 2/5/4 | 0/1/1 |
| rebind_diagonal | 8/8/8 | 7/6/9 | 4/5/5 | 2/2/5 |
| rebind_pooled_outer | 4/4/6 | 4/5/5 | 3/4/4 | 1/3/5 |
| rebind_block_outer | 9/12/7 | 8/8/8 | 6/8/4 | 4/8/5 |

Word-level improvements rule out the blanket description “no new content is read,” but only a correct complete answer establishes successful three-word composition for a case. [answer_diagnostics.json](results/block/answer_diagnostics.json) retains format, length, full-answer containment, fallback to old A/B, and arbitrary training-payload output rates per condition. Neither first-word accuracy nor a slot hit substitutes for EM.

## 4. D confirmation: local gains persist, the full criterion fails

D contains new complete combinations unused in training or development scoring. Each model independently writes D into an A bank, then restores A. **Updated correctness and joint update_restore correctness must remain separate:** rebind block answers 7 D cases correctly for seed 91043, but only 6 also restore A correctly.

| Arm | Known D updated | Known D update_restore | Known D shuffle | Known D empty | Known D locality_joint |
| --- | --- | --- | --- | --- | --- |
| static_diagonal | 0/2/1 | 0/2/1 | 0/0/0 | 0/0/0 | 16/16/16 |
| static_pooled_outer | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 16/16/16 |
| static_block_outer | 2/3/3 | 2/3/3 | 0/0/0 | 0/0/0 | 16/16/16 |
| rebind_diagonal | 3/3/2 | 3/3/2 | 0/0/0 | 0/0/0 | 15/16/14 |
| rebind_pooled_outer | 1/0/0 | 1/0/0 | 0/0/0 | 0/0/0 | 15/16/14 |
| rebind_block_outer | **6/7/3** | **6/6/3** | 0/0/0 | 0/0/0 | 15/15/15 |

In the parameter-matched comparison, rebind block improves D update_restore over pooled by 5/6/3 cases and over diagonal by 3/3/1. New D content for known entities improves, but static block scored 0/0/0 on development C before reaching 2/3/3 on D. Confirmation D cannot retroactively support “consistent improvement on all novel combinations.”

Single-sided complete-answer correctness for new-entity confirmation D is below. CC/HC/CH/HH denote canonical/canonical, held-out support/canonical query, canonical support/held-out query, and both held out, respectively.

| Arm | Confirm CC D | HC D | CH D | HH D |
| --- | --- | --- | --- | --- |
| static_diagonal | 0/1/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| static_pooled_outer | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| static_block_outer | 0/2/0 | 0/2/1 | 0/0/0 | 0/0/0 |
| rebind_diagonal | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_pooled_outer | 1/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_block_outer | **3/1/1** | **1/0/0** | 0/0/0 | 0/0/0 |

New-entity CC update_restore is 0/0/0 for static diagonal,0/0/0 for pooled, and 0/1/0 for block; rebind scores are 1/1/0 for diagonal,1/0/0 for pooled, and 3/1/1 for block. All D shuffle/empty scores are 0 in all four phases. CC joint neighbor correctness for the three static readers is 2/3/2,2/4/1,2/4/2; for rebind it is 10/9/10,10/10/10,10/10/10. Initial new-entity reconstruction is already weak, so low joint correctness cannot be attributed entirely to damage from online updates.

For trained complete payloads, new-entity CC A/B pair scores are 1/0/1 for all three static readers and 8/7/8 for all three rebind readers. Random rebinding helps transfer associations between entities and trained complete content, but novel combinations, especially with held-out question expressions, remain difficult. The full summary retains A/B, update/restoration, locality, and controls for every phase, rather than choosing the best format.

| Arm | Known D edited word correct | Known D both retained words correct |
| --- | --- | --- |
| static_diagonal | 5/3/4 | 6/8/6 |
| static_pooled_outer | 3/4/3 | 6/3/6 |
| static_block_outer | 8/7/4 | 6/7/10 |
| rebind_diagonal | 7/10/8 | 8/8/7 |
| rebind_pooled_outer | 7/7/5 | 4/6/6 |
| rebind_block_outer | 11/11/10 | 9/10/7 |

Low full-answer EM is not merely formatting failure. Among 288 real updated known D generations,287 have exactly three words; answer containment and EM both count 36 correct. Among 1,152 new-entity D generations across four phases,1,110 have exactly three words; containment and EM both count 15. Neither group has 32-token truncation. These are repeated measurements across models, not independent-sample inference.

The preregistered continuation criterion requires **all three seeds** to improve known C and D update_restore over same-seed pooled by at least 4/16 each; achieve known A/B pair≥12/16; exceed both shuffle and empty by at least 4/16 on single-sided real C/D; and have qualified teachers. No static seed passes. Only rebind seed 91043 passes individually:91042 fails C, and 91044 fails C/D. Thus both regimes fail the complete criterion. This is a declared evidence standard, not a statistical-significance test or proof of “no research value.” No corpus expansion occurred, and the best seed is not treated as architectural success.

## 5. Mechanism evidence: an active path, without uniquely attributing gains to rank

Across all 18 models, the independent training audit confirms bitwise-identical shared initialization for all six arms within each seed, and identical added-parameter initialization for all four outer arms within each seed. All 8,256 added coordinates changed in each of the 12 outer models. Added parameters received nonzero gradients on 2,047 steps; step 1 has gradient 0 because b=0, as specified, rather than because the reader is disconnected. See [training_audit.json](results/block/training_audit.json) and [operator_diagnostics.json](results/block/operator_diagnostics.json).

Final Wq/Wk/slot_position tensors are bitwise identical among the three readers within each seed/regime, for all six groups. First-prediction known C fact recall also matches exactly:12/12/13 for static and 15/16/15 for rebind. Reader differences therefore cannot be explained by different learned address parameters. Different generated answers can still change later prefixes and queries, causing decode-routing differences; identical parameters do not imply identical free-generation trajectories. This post hoc check is in [qk_matched.json](results/block/qk_matched.json) and did not affect selection.

At gold-prefix diagnostic positions **before** update 2048, rebind block's added term has ratios `mean(σ₂)/mean(σ₁)` of 0.126118/0.126156/0.126593 and corresponding third-singular-value ratios 0.045665/0.046692/0.047933. The entropy effective rank of this mean spectrum is approximately 1.6515/1.6555/1.6613. In this aggregate sense, it has not completely collapsed to one outer product, although its second and third directions are weaker.

These statistics perform SVD at each position and then average the **ordered spectra of the additional outer term**. They are not the spectrum of an averaged matrix, the mean per-position effective rank, or the full diag+outer spectrum. They are also not free-generation measurements rerun after the final checkpoint. Nonzero secondary singular values or gradients do not establish that useful novel-combination information is carried.

Training CE over the last 128 steps, including EOS with equal sequence weighting:

| Arm | 91042 | 91043 | 91044 |
| --- | ---: | ---: | ---: |
| static_diagonal | 0.016650 | 0.009072 | 0.011087 |
| static_pooled_outer | 0.009743 | 0.008515 | 0.020774 |
| static_block_outer | 0.013861 | 0.008020 | 0.015788 |
| rebind_diagonal | 0.128898 | 0.123798 | 0.117460 |
| rebind_pooled_outer | 0.101995 | 0.093737 | 0.090810 |
| rebind_block_outer | 0.103902 | 0.092485 | 0.085125 |

Outer readers fit rebind training better than diagonal, but pooled and block have similar mean CE while their complete-combination results differ. Equal steps do not guarantee equal optimization, and low CE does not establish successful novel-combination generation. Fixed-budget negative results do not prove a capacity impossibility.

A post hoc development example illustrates the failure boundary: for seed 91042 and known dataset_index=2 (the third record), A is `window lilac eagle` and C is `window lilac badger`. All three static readers retrieve the correct fact throughout decoding yet output old A. Retrieving all three slots does not guarantee reading the edited word. In another rebind example, dataset_index=0 has C=`pencil jade dolphin`: diagonal/pooled output `pencil jade badger`, and block outputs `pencil jade salmon`. All retrieve the correct fact throughout and correct the first word, but corrupt the unchanged third word. These examples were selected post hoc under fixed rules and do not represent the aggregate denominator. The [public examples](results/block/examples.json) retain raw line numbers, source SHAs, complete traces, and selection rules.

## 6. Costs, execution correctness, and reproduction

Completed training totals 36,864 updates,294,912 targets,589,824 sequences,3,059,712 EOS-inclusive gold tokens,44,845,056 actual input positions,46,396,368 padded positions, and 36,864 backbone forwards. Summed training-process time is 3,410.352 seconds. Parallel runs overlap; this is neither exclusive GPU time nor user waiting latency.

Formal student evaluations produced 21,600 generations,106,565 generated tokens,92,502 answer-scoring tokens,4,032 target interventions, and 4,032 restorations, totaling 8,064 real group writes. The teacher produced 448 generations and 2,347 tokens; formal generations total 22,048. Summed generation-process times are 4,718.977 seconds for students and 69.185 for the teacher; they likewise cannot be added and interpreted as exclusive GPU wall time. Initial bank population, control-bank materialization, and smoke runs are excluded from these online logical-write and formal-cost totals.

Answer NLL excludes EOS and is a teacher-forced auxiliary metric, so it is not directly comparable with EOS-inclusive training CE. Per-seed rebind known D token NLL is 1.25623/1.16319/1.50252 for diagonal,2.04645/2.90323/2.55018 for pooled, and 0.75091/0.82072/1.32259 for block. Lower NLL accompanies partial EM improvement but does not establish generalization to complete combinations.

The final [independent CPU audit](results/block/independent_audit.json) passed completely in 116.695 seconds without CUDA. For all 106,565 saved queries across 72 formal student evaluations, it replayed full-bank group selection and complete three-slot retrieval, also checking bank modifications, restoration, controls, and token alignment. Maximum selected-cosine error is approximately 2.98e-7; maximum re-encoded writer key/value errors are approximately 6.16e-7/5.28e-6. All 96 formal jobs are complete with none pending. The separate 6 smoke jobs have 39 generations,117 queries, and 12 group writes and are excluded from formal costs.

The audit did not rerun LM generation or extract the original frozen features again. Query provenance still relies on frozen source and runtime telemetry. Actual queries permit offline verification of full-bank winners, but this is not independent retraining/regeneration of the entire experiment. Uniform weights also do not represent the actual contributions of signed outer factors. The earlier [development audit](results/block/development_audit.json) retains its partial scope; final completeness is established by independent_audit.

Final recomputation commands:

```bash
python scripts/summarize_block.py \
  --runs-root ../runs/xtrah100 \
  --output docs/results/block/summary.json \
  --require-final

python scripts/audit_block.py \
  --runs-root ../runs/xtrah100 \
  --registration docs/results/block/registration.json \
  --output ../plans/block_independent_recheck.json \
  --require-final
```

The second command requires a nonexistent output file to preserve audit receipts.

The generated summary, training_convergence, answer_diagnostics, and key_facts bind their actual source SHAs. The [test receipt](results/block/tests.json) records 1,249 passing full-repository tests, followed by 85 passing registration/new-summarizer tests, plus 14 CPU-audit and 10 training-mechanism diagnostic self-tests. This does not claim a single full rerun of 1,334 tests. Registration and immutable snapshots verify all 49 frozen formal-runtime Python files. Preparation/preflight preceded the final telemetry additions; their own snapshots and receipts remain intact rather than being rewritten as identical versions.

## 7. Next-round proposals, still untested

D preserves a limited development signal but does not meet the full criterion. If mechanism research continues, it should first remain on the same small task and add a matched control that **does not pool raw values first, yet still has rank 1**:

```
M_mean_factors = diag(m) + mean_s[w(v_s)] mean_s[u(v_s)]ᵀ
```

Retain the current u/w definitions without implicitly renormalizing `mean(u)`. Compared with the existing pooled outer product, this separates “pool raw values, then generate factors” from “generate factors per slot, then average.” Its difference from block is exactly the cross-slot covariance term of the factors:

```
mean_s[w_s u_sᵀ] - mean(w) mean(u)ᵀ
  = mean_s[(w_s-mean(w))(u_s-mean(u))ᵀ]
```

This is more identifiable than directly calling a block advantage “rank 3 beating rank 1.” A further mechanism-only diagnostic could hold the selected fact and each factor's marginal set fixed, scramble u/w row pairing, and measure changes in `mean_s[w_s u_perm(s)ᵀ]`. This changes associations; it is not an existing result, and permutations must not be selected using test scores.

These proposals require a new protocol, matched budgets, and independent confirmation. They must not be appended to this round and presented as preregistered arms. This round did not establish that first expanding Wikipedia would solve the problem. The more immediate question is whether nonlinear factor-generation order, within-slot association, and signed coefficients help reconstruct complete content while preserving unchanged words and unrelated facts. Text complexity or scale should be reconsidered only after stable evidence on the small task.
