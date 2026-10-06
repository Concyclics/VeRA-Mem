# Small-data content reconstruction: multi-vector writers and trainable readout

> English localization of the historical report. The archived source, protocol hashes, and registrations remain unchanged; see [reproducibility.md](reproducibility.md) for pinned-source reproduction.

**Status: all formal results are complete and have been strictly recomputed locally.** Under the [fixed protocol](reconstruction_protocol.md) dated 2026-10-06, all 12 training runs, 48 student evaluations, and 5 teacher-qualification jobs completed. All 60/60 formal training/student conditions are present, with no pending or duplicate conditions. The development decision was sealed as “no qualified candidate” at `2026-10-06T12:36:47.921214+00:00`, before any confirmation run started. The [full summary](results/reconstruction/summary.json), [key facts](results/reconstruction/key_facts.json), and [sealed selection](results/reconstruction/selection.json) accompany this report.

The results separate two findings: **the three-slot writer, trainable B, and their combination reliably fit seen A/B, but all four architectures score zero on novel C/D combinations and on new-entity confirmation, failing the writable-memory gates.** The teacher solves every specified condition, so the student's failures cannot be explained by the previous round's unqualified raw-Wiki teacher. The predefined rule stops expansion to 256/1024 records. Trainable B is one constrained readout intervention, not a representative of every stronger reader.

## What this round tests

The previous cold-start experiment produced no complete multiword answers, and some Wiki teacher conditions were unqualified. This round uses an independent miniature synthetic dataset to establish whether reconstruction is learnable, then compares writer and readout. It is not a single-factor repetition with merely less data: targets, key features, loss, and exposure per fact all changed. Cross-round scores cannot estimate the effect of one factor alone.

The specified revision of Qwen3-4B-Instruct-2507 is frozen, with a layer-20 VeRA interface of rank/key dimension 64. Training uses A/B versions of 16 three-word notes. Inference questions contain neither source text nor answers; writes read the observed note. The foundation vector bank is fully disabled, and evaluation uses a real CPU VDB. The teacher only checks task feasibility with raw context and supplies no distillation loss.

| Architecture | Dynamic record | B | Trainable parameters | Vector bytes per fact |
| --- | --- | --- | ---: | ---: |
| S1_fixedB | Payload mean, one slot | Fixed random | 1,870,336 | 512 |
| S3_fixedB | Three word-span means, three slots | Fixed random | 1,870,336 | 1,536 |
| S1_trainB | Payload mean, one slot | Same initialization, trainable | 2,034,176 | 512 |
| S3_trainB | Three word-span means, three slots | Same initialization, trainable | 2,034,176 | 1,536 |

Same-seed arms share initial A/B/Wq/Wk/Wv/b tensors. Centering uses only each writer's training A/B features; S1 and S3 statistics buffers need not match. Three slots share projection parameters rather than using separate writers. Top-4 selects **slots**, so three slots jointly change content granularity, key addressing, and candidate competition. Learning B exposes another 163,840 parameters while rank remains 64; this does not increase residual rank. Vector-byte counts exclude IDs, timestamps, indices, and Python objects.

Optimization seeds are `71042 / 71043 / 71044`, with data seed `101042`. Each model receives 1,024 updates with 8 targets per step, each with independent A/B banks. The resulting 16 student sequences share one batched backbone forward. Loss is full-answer CE including EOS plus `0.2 × fact-group address CE`; both average valid prediction positions within each sequence, then weight sequences equally. Each fact is a target 512 times. Step-512 diagnostics do not select checkpoints; all results use the step-1,024 endpoint.

## Data and scoring boundaries

Training sees A/B for only 16 entities. C is endpoint development data and D is sealed until confirmation; dev has 32 new entities and confirm has 64. Component words occur in training, but the corresponding complete payloads do not. **C/D retain A's first two words and substitute another seen third word.** Novel-combination conclusions therefore concern a new final word after an old prefix, not arbitrary-position recombination or unseen vocabulary. Tensor auditing confirms that all 16/16 C third-word features actually change, while the first two word features remain bitwise equal to A.

Frozen features for all worlds, including D, were encoded before scoring. What was sealed was D's free generation, scoring, and use in selection; D was not wholly uncomputed. D contributes neither gradients nor fitted statistics nor selection. Moving from known 16 to dev 32/confirm 64 jointly changes entities, complete combinations, and bank size. Failure there cannot be uniquely attributed to entity generalization. Same-entity C/D, with the same bank size and canonical format, provides evidence against expansion without those three confounds.

`CC / HC / CH / HH` combine canonical/heldout support with canonical/heldout query; the first letter denotes support. Each target starts from an independent A bank and updates only its own complete slot group. B, C, D, and SWAP do not accumulate. SWAP writes a fixed other entity's A content into the target while leaving the donor unchanged, testing whether seen content can be reassigned. Restoring A recovers keys/values while legitimately increasing the target timestamp, so the whole-bank hash need not return to its original value.

The primary measure is complete-answer EM under free generation. A/B pair requires both worlds correct; C/D update+restore requires correctness after the new-content write and after restoring A. Each update also queries one fixed neighbor `(i+1)%N`. `locality_joint` requires correctness before and after; an unchanged wrong answer is not successful preservation.

Shuffle permutes values only, retaining keys and moving complete three-slot fact groups together. Empty removes all dynamic memory. C/D comparisons use single-world EM so that the logically zero paired EM of a deterministic empty bank facing different A/B answers is not mistaken for evidence. R@4 means at least one of four selected slots belongs to the target fact; it does not imply complete content readout or four distinct retrieved facts.

## Development results and the sealed decision

Triples below follow seed order `71042 / 71043 / 71044`, with denominator 16 each. New-entity development uses denominator 32. Repeated worlds and seeds are not pooled into independent facts.

| Architecture | Seen A/B pair | C write + A restore | C neighbor correct at both times | New-entity dev CC pair |
| --- | --- | --- | --- | --- |
| S1_fixedB | 14 / 13 / 13 | 0 / 0 / 0 | 15 / 14 / 14 | 0 / 0 / 0 |
| S3_fixedB | 16 / 16 / 16 | 0 / 0 / 0 | 16 / 15 / 15 | 0 / 0 / 0 |
| S1_trainB | 16 / 16 / 16 | 0 / 0 / 0 | 16 / 15 / 16 | 0 / 0 / 0 |
| S3_trainB | 16 / 16 / 16 | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 |

The three structural arms also achieve 16/16 in every seed for joint correctness at all A→B→A time points; the baseline achieves 14/13/13. Seen-B single-world shuffle and empty EM are 0/16 for every model. Training answers therefore are not produced unchanged under these controls. This establishes dependence on written vectors for seen-label reconstruction, not that the vectors carry compositionally usable new content.

Every model has C real, shuffle, and empty EM of 0/16, so real-control differences are zero. The structural arms improve seen reconstruction, but none satisfies G1/G2 in every seed: A/B pair at least 15/16, C update+restore at least 15/16, C real exceeding both controls by at least 50 percentage points, and C neighbor joint correctness at least 95% (16/16). The sealed decision is `selected_arm = null`, not a choice of the relatively best failing model.

High C neighbor preservation must not hide another problem: **writing trained B still interferes with an unchanged neighbor.** B neighbor joint correctness is 8/8/5 for baseline, 12/11/13 for S3_fixedB, 9/11/11 for S1_trainB, and 14/15/14 for S3_trainB, all out of 16. Correct target answers and locality are different measures.

SWAP single-answer EM is 0/0/0 for baseline, 0/0/0 for S3_fixedB, 1/0/0 for S1_trainB, and 1/1/2 for S3_trainB, all out of 16. Reassigning already seen content is also difficult, beyond C's unseen complete combination. This is not an independent elimination of all addressing explanations: the update changes target keys and values together.

## Teacher qualification and error diagnostics

The teacher checks feasibility with raw text context; this round has no KD. All 784 teacher generations are fully correct (784/784): preflight train A/B each 16/16; development known CC A/B/C and HC/CH/HH A/B each 16/16; dev CC A/B each 32/32; final known CC A/D each 16/16; and new-entity confirm A/B in all four phases each 64/64. Every corresponding paired qualification passes, so eligible subsets equal full denominators. SWAP has no additional teacher job.

C first-prediction R@4 is 16/16 in 11/12 models and 15/16 for S3_trainB seed 71044, yet full EM is zero throughout. Multi-slot models sometimes preserve the first two words but fail to produce the new third word. This rules out “every failure is a complete first-step target miss,” but **does not establish a purely reader-related cause**. Fact R@4 needs only one target slot; it guarantees neither sufficient weight on the relevant word slot nor continued useful reads at later prediction positions. D first-prediction R@4 has the same pattern: 11 models at 16/16 and S3_trainB seed 71044 at 15/16. Initial recall, decode residence, and content correctness remain distinct.

New-entity dev CC A and B single-world EM are also zero for every model. Known CH/HH pair is zero throughout; HC has only 1/16 and 4/16 for S3_trainB seeds 71042 and 71044, with all others zero. New formats and new entities are different distribution shifts and should not be collapsed into one generalization score.

Independent raw-text recomputation gives 16 shared facts × 4 architectures × 3 seeds = 192 predictions for each of C and D, not 192 independent facts. C has both full EM and complete-answer containment of 0/192; 187 outputs contain exactly three words and none hits the token budget. D likewise has 0/192 on both measures; 188 outputs contain exactly three words and only 1 reaches the 32-token limit. Zero accuracy cannot be explained solely by extra formatting or widespread truncation. The first two words are both correct in 127/192 C and 136/192 D outputs, but the third word is correct in only 1/192 and 3/192.

[Post-hoc content and routing diagnostics](results/reconstruction/content_diagnostics.json) cover all 24 C/D conditions and are not used for selection or tuning. Each row below represents 48 repeated predictions of the same 16 facts across three seeds:

| Architecture / world | Full output equals old A (/48) | Third-word slot retrieved at first prediction (/48) | Third-word slot retrieved anywhere (/48) | Decode third-word slot hits/queries |
| --- | ---: | ---: | ---: | ---: |
| S3_fixedB / C | 43 | 11 | 33 | 67/199 |
| S3_fixedB / D | 42 | 17 | 37 | 61/199 |
| S3_trainB / C | 43 | 11 | 32 | 79/211 |
| S3_trainB / D | 43 | 21 | 40 | 83/205 |

High fact R@4 often covers only slots corresponding to the old prefix. However, many D generations do retrieve the third-word slot at some point and still fail, so failures cannot all be described as “the third word was never retrieved.” These traces lack generated token IDs, preventing exact alignment of a slot hit with the third-word prediction. A hit also does not establish sufficient contribution. Repeating old A describes outputs; it does not identify whether the old label resides in shared parameters, other slots, or subsequent language-model dynamics, nor independently establish causal reader failure.

Evaluation NLL averages answer tokens under a gold prefix and excludes EOS; training CE includes EOS and equally weights sequences, so the measures are not directly comparable. C NLL ranges from 3.009–4.219 and D from 2.918–3.827 across models, while complete free-generation EM remains zero. Better NLL does not replace writability.

## Final confirmation and expansion decision

“No candidate” was sealed before any confirmation began. Under the fixed rule, this round is ineligible for expansion to 256/1024 records. Confirmation still reports sealed D combinations and new entities; its results cannot retrospectively change the candidate.

| Architecture | D write + A restore (/16) | D neighbor joint correctness (/16) | New-entity confirm CC pair (/64) | HC / CH / HH pair (each /64) |
| --- | --- | --- | --- | --- |
| S1_fixedB | 0 / 0 / 0 | 15 / 14 / 14 | 0 / 0 / 0 | 0 for every seed and phase |
| S3_fixedB | 0 / 0 / 0 | 15 / 15 / 15 | 0 / 0 / 0 | 0 for every seed and phase |
| S1_trainB | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 | 0 for every seed and phase |
| S3_trainB | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 | 0 for every seed and phase |

D single-world real/shuffle/empty EM is also 0/16 throughout. Restored A is 15/16 in each baseline seed and 16/16 in each structural-arm seed. New-entity confirmation A and B single-world scores are each 0/64 in every phase, so zero pairs do not merely reflect one failing side. G3 requires the preselected architecture to attain D update+restore ≥15/16 and confirm CC pair ≥52/64 separately for all seeds. There is neither a development-qualified candidate nor a passing confirmation result. In the summary, `final_gate.per_seed.complete=false` means there is no preselected architecture to assess; **it does not mean formal confirmation is missing**. `all_formal_results_complete=true`.

## Cost and audit scope

All 12 formal training runs have been recomputed: 12,288 updates, 98,304 target exposures, 196,608 A/B student sequences, 1,019,904 gold tokens, 14,954,496 actual input positions, 15,481,152 padded input positions, and 12,288 backbone forwards. Recorded training-process durations sum to 1,113.422 seconds. This is cumulative process time, not exclusive GPU latency or launch-to-completion wall time.

Students generated 11,904 outputs and 61,190 tokens, with another 49,920 answer-scoring tokens. There were 4,800 independent target interventions and 768 restorations, totaling 5,568 real group writes, matching the plan. The teacher generated another 784 outputs and 4,069 tokens. Summed student-generation time is 2,382.482 seconds and teacher-generation time 115.735 seconds. These cover only their timed intervals, exclude complete scoring/initialization/backup, and can overlap in parallel. Initial bank population and shuffle/empty materialization are not counted as online single-record writes. Explicit smoke runs are excluded from formal training statistics.

The strict summarizer recomputes EM, pairs, restoration, locality, and R@4 from raw predictions and checks world/phase/trigger, labels, generation counts, target schedules, loss, budgets, checkpoint lineage, and file SHAs. It does not deserialize model/bank tensors. The [independent audit](results/reconstruction/independent_audit.json) separately re-encodes writer K/V from cached features and replays saved CPU single-record writes/restorations. Same-seed initialization and endpoint frozen/Adam states also pass checks.

The independent audit covers 66 formal jobs, including 1 prepare, with 12,688 formal student-plus-teacher generations. Another 7 smoke jobs contain 39 generations, 6 interventions, and 6 restorations, excluded from formal scores/costs. The audit ran on CPU for 68.994 seconds without initializing CUDA, with `complete=true` and `checks_passed=true`. It **does not rerun language-model generation or query logits**. Writer re-encoding starts from saved frozen features rather than running the backbone over raw text again. Logged frozen flags are not trusted-execution proofs, and saved initial/final tensors cannot prove an absence of gradients at every intermediate moment. Hash checking, tensor replay, and actual model reruns provide different evidence.

A concurrent clarification caused one paragraph of the baseline source-snapshot protocol to differ from the registered document: registered SHA `38ff9df8681757dd963e2cb4b63405c7f306d887b8b0442f7db823625896e7e5`, baseline copy `2cf86a9527907bdf215407249a78834a1aac2f685ceeb078a5712d766a24ba5c`. The difference only clarified that C/D retain the first two words and replace the third, and their selection roles. Training Python sources matched registration file by file; data, budget, and gates did not change. Immutable run copies were preserved, and the live protocol was restored to its registered version at the end of that experiment. See the [protocol-copy note](results/reconstruction/protocol_copy_note.json) for the line-level diff and checks. This document-copy difference is not a model/data version change. The [registration](results/reconstruction/registration.json), [training-matrix audit](results/reconstruction/training_matrix_audit.json), and [data-provenance audit](results/reconstruction/data_provenance_audit.json) preserve sealing, initialization/exposure, and split evidence.

## Recompute

From the repository root, use the fully backed-up local runs; no model, GPU, or SSH is required. Read the sealed selection without rewriting it:

```bash
python3 scripts/summarize_reconstruction.py \
  --runs-root ../runs/xtrah100 \
  --output ../plans/reconstruction_summary_final_20261006.json \
  --selection ../plans/reconstruction_selection_20261006.json
```

The original selection command used the same script's `--select --selection-output ...` mode. It requires all 12 training and 24 development evaluations and no existing confirmation child. Do not rerun that mode after confirmation starts. Summarizer tests are `python -m pytest tests/test_summarize_reconstruction.py -q`.

The full pytest run passed 947 tests in 99.34 seconds. A later 93-test confirmation-scheduler hardening run passed; the final collection contained 989 tests, without claiming that all 989 were rerun together. Independent-audit self-tests passed 19 cases and content-diagnostic self-tests passed 6. All 40 registered `src/*.py` files still matched their original SHAs, and the canonical protocol retained its registered version in the historical source.

Formal source, configuration, data, and commands are defined by registration, suite source snapshots, manifests, and [historical plans](results/reconstruction/plans.json). Public [summary.json](results/reconstruction/summary.json) is byte-identical to the local strict result, with SHA-256 `050d5497d1dd5f82f820dd23f164a3960261908826ad7455ed37756511735282`. [key_facts.json](results/reconstruction/key_facts.json) retains that source SHA, per-seed integer counts, and independent raw-output recomputation for C/D.

## Limits and next directions

Under this recipe, structural changes enable stable free-generation reconstruction of 16 seen A/B facts. They do not establish the ability to answer novel combinations through online writes alone, and seen-B writes can interfere with neighbors. This gap neither disproves vector memory in principle nor proves that fixed rank 64 is its cause.

The three seeds share a miniature dataset and templates; they measure initialization/optimization stability, not three independent datasets. Three slots change writer granularity and addressing at three times the vector bytes per fact; learning B increases trainable capacity. Equal updates do not mean equal FLOPs, time, or parameters, so no universal efficiency advantage is established. C/D modify only the final word, and each update probes only one neighbor, limiting composition and locality claims respectively.

If work continues, first test whether content can be decoded through the interface on newly sealed miniature data, rather than adding a large corpus or treating recall/NLL alone as success. A positive control separating content-decode learnability from unrestricted addressing could precede testing new combinations with shared parameters frozen. Any forced target-record read or word-level supervision must declare its extra access/labels rather than masquerade as a result from this round. New structure, supervision, and gates require another protocol; this round adds neither training nor corpus expansion.
