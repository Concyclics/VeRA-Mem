# Code map

Module families retain their original names so historical checkpoints, plans, and source snapshots remain traceable. Organization is provided through this map and the documentation index rather than moving recorded entry points.

## Start with the current method

| File | Responsibility |
| --- | --- |
| [`block_vera.py`](../src/vera_mem/block_vera.py) | Full-group retrieval, three reader operators, block-aware persistent store |
| [`block_backend.py`](../src/vera_mem/block_backend.py) | Carries the actual query and complete selected V rows across the CPU/GPU hook boundary |
| [`block_data.py`](../src/vera_mem/block_data.py) | Controlled entity/payload splits, static/random bindings, and episode schedules |
| [`block_run.py`](../src/vera_mem/block_run.py) | Feature preparation, initialization, optimizer groups, training, and stage CLI |
| [`block_eval.py`](../src/vera_mem/block_eval.py) | Teacher qualification, frozen online interventions, generation, route traces, and restoration |
| [`test_block_vera.py`](../tests/test_block_vera.py) | Operator, retrieval, gradient, snapshot, and architecture contracts |
| [`test_block_run.py`](../tests/test_block_run.py) | Initialization, feature, schedule, and training-interface contracts |

`BlockVeRA` inherits the learned address/value interface from `QKVVeRA`, which builds on the reconstruction and stable-vector families. Before changing a superclass, check dependent families and checkpoint state validation. Keep architecture changes explicitly versioned; do not load an old checkpoint silently into a different reader.

## Family index

| Family | Runtime/data/evaluation entry points | Main report |
| --- | --- | --- |
| Initial vector memory | `vector_vera.py`, `vector_store.py`, `vector_run.py`; `backend.py` | [Pilot](pilot_results.md) |
| Baselines | `run.py`, `memory.py`, `metrics.py`, `data.py`; `configs/baselines*.json` | [Pilot](pilot_results.md) |
| Stable/scaling | `stable_vector_vera.py`, `scaling_backend.py`, `scaling_run.py` | [Scaling](scaling_results.md) |
| Augmentation/address probes | `augmentation_data.py`, `generalization_run.py`, `factcentric_losses.py` | [Generalization](generalization_results.md), [fact-centric](factcentric_training.md) |
| Context distillation | `context_distillation.py`, `context_distillation_run.py` | [Context KD](context_distillation_results.md) |
| Counterfactual KD | `counterfactual_data.py`, `counterfactual_losses.py`, `counterfactual_backend.py`, `counterfactual_run.py`, `counterfactual_eval.py` | [Counterfactual](counterfactual_results.md) |
| Interface decomposition | `interface_data.py`, `interface_features.py`, `interface_writer.py`, `interface_variants.py`, `interface_losses.py`, `interface_training.py`, `interface_run.py`, `interface_eval.py`, `interface_diagnostic.py` | [Interface](interface_results.md) |
| Foundation dictionary | `dictionary_vera.py`, `coldstart_data.py`, `coldstart_run.py`, `coldstart_eval.py`, `coldstart_teacher.py` | [Cold start](coldstart_results.md) |
| Reconstruction | `reconstruction_vera.py`, `reconstruction_data.py`, `reconstruction_run.py`, `reconstruction_eval.py` | [Reconstruction](reconstruction_results.md) |
| Grouped QKV | `qkv_vera.py`, `qkv_data.py`, `qkv_run.py`, `qkv_eval.py` | [QKV](qkv_results.md) |
| Whole-block operator | `block_vera.py`, `block_backend.py`, `block_data.py`, `block_run.py`, `block_eval.py` | [Block](block_results.md) |

All runtime files are under `src/vera_mem/`; each family has corresponding `tests/test_*.py` coverage. Only the initial families use the top-level `configs/` files. Later controlled studies express settings in stage CLIs and explicit job plans.

## Orchestration and evidence

| Scripts | Role |
| --- | --- |
| `prepare_model.py`, `prepare_data.py`, `prepare_*_plans.py` | Pinned assets and explicit plans; later historical plan generators are host-specific |
| `run_*_suite.py`, `run_*_eval_suite.py` | Execute plans into fresh run directories with source snapshots and exit status |
| `launch_*.py`, `orchestrate_*.py`, `interface_transport.py` | Original remote scheduling and SSH transport; inspect machine assumptions before reuse |
| `seal_block.py`, `block_registry.py`, `confirm_qkv.py` | Registration and execution barriers; do not bypass validation to reuse historical names |
| `summarize_*.py` | Strict result aggregation and report rendering; English output in the current checkout |
| `audit_*.py`, `replay_coldstart_banks.py` | Evidence and persisted-bank checks; scope varies by study |
| `diagnose_*.py`, `probe_*.py`, `plot_*.py` | Mechanism, error, feature, and visualization diagnostics |
| `sync_backup.py` | Verified artifact/source mirror excluding base weights and environment caches |
| `check_repository.py` | English-content, document-link, and localization-integrity check |

## Where to extend the method

For a new rank-one factor-mean control, begin with the operator in `block_vera.py`, but create an explicit new architecture mode/version and separate experiment protocol. Update checkpoint validation, backend contracts, operator/gradient tests, trainable-parameter accounting, and plan registration together. Reusing the current `block_outer` label for a changed formula would make historical results ambiguous.

For new query or support formats, start in the data family and feature preparation. Verify token/span alignment and teacher qualification before expensive training. Keep query input free of support/gold-answer text, and do not expose direct fact IDs to retrieval.

For scaling to multiple retrieved fact groups or approximate search, treat both as new experimental axes. Measure interference, retained content, CPU/GPU transfer, and retrieval error independently of the reader comparison. The present exact-search implementation is deliberately small.
