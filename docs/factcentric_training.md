# Fact-centric training: an initial test of reduced format interference

Existing evidence supports **format sensitivity in layer-input representations and Q/K/value encoders**. It does not attribute all failure to VDB similarity search, or prove that the representations lack semantic information. This round implemented three targeted training probes: all-view training improves addressing from original questions to new observation formats, and a fixed canonical teacher improves value compatibility with the original vector coordinates. New question forms remain unresolved. These are encoder experiments on cached features, **not generation-accuracy results, and they do not replace the usable same-template checkpoint**.

Full summary: [probe.json](results/factcentric/probe.json). Earlier end-to-end experiments: [generalization_results.md](generalization_results.md). Representation diagnostics: [address](generalization_address_diagnostic.md) and [value](generalization_value_diagnostic.md).

## Interpreting style-sensitive retrieval

VDB search operates in the encoders' cosine space. If queries and keys do not preserve fact identity across formats, changing the database implementation will generally not repair that representation problem. Earlier diagnostics found that training-template means explain 94.06% of centered query-input energy. This does not mean that 94.06% of retrieval decisions are caused by style, nor is it a measurement of semantic information. Removing the training-template subspace improves some new-observation retrieval but leaves new questions poor, indicating that format interference is only part of the problem.

Current augmented training already samples query and support formats independently; it does not directly reward matching styles as positives. Other possible factors include poorly conditioned representations, insufficient training-view coverage, limited updates, and writer/reader coordinate drift. A random entity-ID→16-word task tests fact identity and association retention, not open-domain semantic understanding.

Training must constrain two things simultaneously: different expressions of the same fact should address one another, and different facts must remain distinguishable even when their format or answer is identical. Values must also be readable by the existing VeRA branch. Bringing two trainable views closer together alone does not specify what information they should retain.

## Implementation and runs

Use fixed Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`, with cached layer-20 `mlp.down_proj` inputs and rank/key dimensions of 64. All three students initialize Wq/Wk/Wv and A/B/b from the earlier canonical4096 checkpoint at step 1536. Query/support centers are refitted using all training views. Shared b and random matrices are frozen; only Wq/Wk/Wv are trained.

There are 4096 training facts, each with 8 question views and 4 observation views. Each arm receives 1600 Adam updates, batch size 128, learning rate 1e-4, and gradient clipping at 1. Each batch includes 8 distinct facts for each of the 16 answer labels. Labels only construct same-answer/different-entity negatives and are not inputs to the query encoder. All three arms use exactly the same fact-exposure sequence, with 204,800 exposures each. The final step is fixed; development data do not select checkpoints or hyperparameters.

| Method | Encoded views per fact | Actual objective |
| --- | --- | --- |
| sampled_pair | 2 queries, 2 supports | Bidirectional entity CE on one random view pair, plus 0.1×paired Q/K/value consistency |
| all_view | 8 queries, 4 supports | Bidirectional entity CE separately for every query-style×support-style pair, averaging 32 combinations equally; the same consistency term |
| anchored_all_view | Same as all_view | all_view plus Q/K cosine alignment and RMS value MSE to a fixed canonical teacher; all three anchor weights are 1 |

Within each style combination, all candidate keys have the same observation format, so a format preference cannot identify the correct entity. All combinations enter the loss, rather than allowing only the easiest positive view to match. Teacher targets are generated solely from **canonical-template inputs of training facts**, with stopped gradients. Neither development nor confirmation entities are used for training.

Code: [factcentric_losses.py](../src/vera_mem/factcentric_losses.py) and [probe_factcentric_training.py](../scripts/probe_factcentric_training.py). The loss library also provides uniform multi-positive CE over flattened views. **The actual probe uses `style_block_retrieval_loss`, with a separate bank for each style combination**; these are not the same objective. The latter does not additionally calibrate logits across differently formatted banks, so mixed-format bank evaluation is still needed.

Fact counts and update counts match across all three arms, but **computation does not**: sampled_pair encodes 409,600 queries and supports each; each all-view arm encodes 1,638,400 queries and 819,200 supports. Only all_view and anchored_all_view match view counts exactly. H100 training and diagnostics take approximately 13.62, 15.40, and 16.68 seconds, respectively. These exclude cache construction, teacher training, and shared preparation, and are not end-to-end costs.

## Development results

The development set contains 64 entities unused in training. Every cell is **R@1 hits / 64**, crossing three question forms with three observation formats. The nine conditions reuse the same entities and are not nine independent samples. These development formats were inspected in earlier diagnostics, so this round is exploratory rather than a new confirmation experiment.

| Query → support | sampled_pair | all_view | anchored_all_view |
| --- | ---: | ---: | ---: |
| Original question → original observation | 63/64 | 64/64 | 64/64 |
| Original question → new observation 1 | 31/64 | **39/64** | 38/64 |
| Original question → new observation 2 | 25/64 | **38/64** | 36/64 |
| New question 1 → original observation | 3/64 | 5/64 | 3/64 |
| New question 1 → new observation 1 | 2/64 | 3/64 | 9/64 |
| New question 1 → new observation 2 | 2/64 | 2/64 | 3/64 |
| New question 2 → original observation | 7/64 | 2/64 | 3/64 |
| New question 2 → new observation 1 | 3/64 | 2/64 | 2/64 |
| New question 2 → new observation 2 | 2/64 | 1/64 | 1/64 |

New question forms 1/2 are held-out request-card and leading-constraint formats; exact text is defined in the cache protocol. All-view training improves original-question→new-observation retrieval from 48.44%/39.06% to 60.94%/59.38%. This is a local addressing improvement; one seed and 64 entities cannot establish a stable effect. The fixed teacher does not improve addressing uniformly, and new-question conditions remain at only 1–9/64.

We also evaluate compatibility with teacher coordinates using **16 answer prototypes obtained by averaging training-set teacher values by label**:

| Observation format | sampled_pair | all_view | anchored_all_view |
| --- | ---: | ---: | ---: |
| Original observation | 33/64 | 26/64 | **64/64** |
| New observation 1 | 12/64 | 7/64 | 19/64 |
| New observation 2 | 11/64 | 9/64 | 16/64 |

Mean cosine between canonical values and the same-fact teacher values is 0.456, 0.411, and 0.989, respectively. Anchoring can therefore preserve existing coordinates. However, Wv in the first two arms receives **only consistency loss, without LM or answer supervision**, whereas the third receives explicit teacher-value targets. This is not a fair comparison of answer-semantic learning. It does not establish that the first two lose semantics or that the third improves generation. To measure information retained in each student, fit answer prototypes or a linear diagnostic head from that student's own training values.

This round does not run Qwen forward training, token-wise decoding, actual VeRA residual readout, online sequential writes, or answer EM. It also does not select or evaluate confirmation features. Earlier end-to-end negative results remain unchanged.

## Proposed next training stage, not yet run

Retain fact-level view combinations and entity negatives, while restoring an LM loss that constrains what values are useful for. Validate addressing and readout separately:

```text
L = L_LM(real retrieved values)
    + λ_addr L_style_block_entity
    + λ_anchor (L_Q_teacher + L_K_teacher + L_value_teacher)
    + λ_token L_answer_position_address
    + λ_replay L_LM(canonical replay)
```

1. **Increase structural variation, not just vocabulary variation.** Cover field order, entity position, question/record formats, length, and irrelevant context. Pair multiple views of a fact with same-format negatives from other facts. For real multi-relation data, define positives by full fact identity—entity, relation, and time/version—not just entity or answer.
2. **Fix reliable targets while training real readout.** Retain the canonical teacher and some canonical replay; jointly train LM behavior, addressing, and value alignment to prevent writer/reader co-drift. The teacher supplies validated same-template coordinates, not semantic coverage of unseen formats by itself.
3. **Supervise addressing at positions that can observe fact identity.** Use the final prompt token and answer-prediction positions. Do not force all tokens before the entity appears to retrieve that entity. Actual inference still generates a query from each token's VeRA-layer input; an external entity parser does not replace the main path.
4. **Keep evaluation discriminating.** Compare sampled/all-view/anchor under fixed budgets while reporting extra encoding and cache costs. A new confirmation set should isolate entities and structural styles and include mixed-format banks, real retrieval/correct-address oracle/shuffled values/zero increment, post-update retention, and multiple seeds. The old confirmation set has already informed previous analyses; the next round needs an unseen set.

If new question forms still cannot address facts reliably after broader training views and joint LM supervision, separately test nonlinear encoders, other layers, or input aggregation. Do not combine structural and loss changes in one comparison, or judge solvability solely by continually increasing consistency weight.

## Basis, reproduction, and audit

- [Khosla et al., Supervised Contrastive Learning, NeurIPS 2020](https://arxiv.org/abs/2004.11362): motivates joint supervision from multiple positive views of one identity. Here, positives use fact identity and the actual probes use style-specific banks. The paper's classification conclusions are not transferred directly to VeRA.
- [Romero et al., FitNets: Hints for Thin Deep Nets, ICLR 2015](https://arxiv.org/abs/1412.6550): fixed teacher intermediate representations can supervise students. Cross-format Q/K/value anchoring within the same backbone is a research hypothesis inspired by that idea, not a memory-generalization result established by the paper.
- [Bardes et al., VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906): view consistency alone permits uninformative solutions, requiring mechanisms that preserve discriminating information. This round uses entity negatives and a fixed teacher; it does not implement VICReg or claim a theoretical non-collapse guarantee.

To reproduce in a workspace with the existing cache and teacher, first check available resources and set `GPU_UUID` to an available GPU:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=4 \
PYTHONPATH="../env_deps:src" python scripts/probe_factcentric_training.py \
  --device cuda \
  --cache ../data/generalization/features_v1.pt \
  --teacher ../runs/generalization_control_20261005/canonical4096/best.pt \
  --output ../runs/factcentric_probe_reproduction/metrics.json
```

The run identifier is `factcentric_probe_20261006`; the actual start time was **2026-10-05 18:30 UTC**. Directory names are not timestamps. The process completed with exit 0, and all three checkpoints and source hashes were verified. Public JSON retains data/model/source/weight hashes and training history. Raw logs, executed source snapshots, and full module checkpoints remain in the experiment workspace and are backed up locally; model weights are not committed to the code repository.

Cache SHA256: `00b94acb30450f31285c5268bb5709485a185c81327cf0dabcf02c1e1c66ab32`. All three fact-exposure SHA256 values are `f2412f71484a9acadc085acac64d314b1e9e75e6d2740b0fb2a2c03f5b6483c7`. All 140 local tests passed, including 15 new tests for losses, stopped gradients, entity negatives, sampling, and encoding equivalence. Those 15 new tests also passed on CPU in the H100 server's Python environment.
