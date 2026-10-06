# Collaborating on VeRA-Mem

Start with the [research path](docs/research_path.md), [architecture](docs/architecture.md), and [latest complete results](docs/block_results.md). The repository is a reproducible research prototype with a limited positive block-readout result, not a finished continual-learning system.

## Current research objective

Learn a writer/reader interface that can store, revise, and retrieve previously unseen content through forward-pass K/V writes while keeping shared model parameters fixed online. Success requires full answers to follow the current memory, correct restoration, preserved unrelated facts, and transfer across entities and expressions.

Do not collapse these distinctions:

- New entity versus new entity/content binding versus a new complete content combination.
- Better retrieval versus better full-answer generation.
- Seen reconstruction versus online writing of unseen content.
- Gold-prefix likelihood versus free-generation correctness.
- A changed answer versus a correct change in both worlds.
- Within-format behavior versus heldout-query and heldout-support transfer.

## First proposed comparison

Use the same diagonal residual and factor definitions, with three outer terms:

```text
Pooled:       w(mean(V)) u(mean(V))^T
Factor mean:  mean_s(w(v_s)) mean_s(u(v_s))^T
Block:        mean_s(w(v_s) u(v_s)^T)
```

The factor-mean control is rank one but computes each nonlinear input factor before averaging. Do not renormalize its averaged `u` implicitly. Its difference from the block term is the covariance of the two factors across rows. A separate diagnostic may shuffle row pairings between `w` and `u` while preserving the selected group and marginal sets.

These controls have not been tested. Their purpose is to distinguish nonlinear processing order, rank, and row association. Any temperature, normalization, gating, or width change should be a named additional axis rather than an unreported improvement to one arm.

## Prepare a new experiment

1. Write the hypothesis, arms, endpoint metrics, budgets, and continuation rule before inspecting fresh confirmation outputs. Freeze new entity/content splits; the current C/D sets are already observed.
2. Match permitted information, task schedules, initialization, and storage across arms. Report unmatched parameter counts, FLOPs, feature exposure, and optimization work explicitly.
3. Qualify the context-provided teacher on training/development formats. Verify support spans, writer prefixes, and actual contextual re-encoding for binding changes.
4. Run a small pipeline check and relevant CPU tests. Check available compute and disk; use only allocated or authorized resources.
5. Create new plan, registration, output, and checkpoint identities. Preserve source snapshots and strict evidence barriers. Do not overwrite historical runs or alter their hashes.
6. Train and evaluate all prespecified arms/seeds. Any exploratory change after seeing results becomes a new experiment with its own confirmation set.

## Required measurements

Report full-answer EM for A/B, novel C/D content, new entities, and canonical/heldout support/query formats. Keep explicit denominators and seed columns. Pair actual writes with restoration and unrelated-fact checks; include real, shuffled-value, and empty-bank controls. Teacher-qualified subsets complement, rather than replace, the full sample.

Record actual queries, selected group/rows, scores, weights, generated token IDs, bank snapshots, write events, checkpoint/source hashes, and encoder version. Log optimizer updates, target exposures, unique facts, gold/generated tokens, input positions, timing, memory bytes, and shared parameters. A CPU evidence replay is not an independent language-model rerun.

Keep unsuccessful runs and exclusion reasons. Bootstrap over facts/entities with shared views kept together; repeated model/format measurements are not independent facts. A single training seed cannot establish robustness across optimization randomness.

## Code and documentation

Keep current documentation, comments, help text, report output, and explanatory artifact strings in English. Mathematical symbols and non-ASCII Unicode test coverage are allowed; user-facing Chinese text is not part of the current handoff. Do not alter raw observed outputs merely to change presentation language.

Use the [code map](docs/code_map.md) to locate the relevant module family. Add meaningful tests when changing storage, gradient flow, retrieval, data boundaries, or checkpoint compatibility. Run affected tests and `python scripts/check_repository.py`; use a full suite when shared interfaces change. Avoid refactoring archived mechanisms solely for cosmetic uniformity.

Before sharing a result, link the protocol, code revision, plans, summary, audit scope, and cost record. Never present a translated protocol as the original sealed bytes. See [reproducibility](docs/reproducibility.md) for historical audits and current English provenance.
