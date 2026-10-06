# Documentation index

Read [research_path.md](research_path.md) first for the experimental narrative and claim boundaries. Then use [architecture.md](architecture.md), [code_map.md](code_map.md), and [reproducibility.md](reproducibility.md) to work with the implementation. Proposed next experiments are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Experiment archive

| Stage | Protocol/design | Results | Additional evidence |
| --- | --- | --- | --- |
| Initial method and pilots | [Literature/design](literature_and_design.md), [data protocol](data_protocol.md), [memory contract](vector_memory_contract.md) | [Pilot results](pilot_results.md) | [Initial](results/initial/report.md), [centered](results/centered/report.md), [staged](results/staged/report.md) |
| Scaling and initialization | [Memory contract](vector_memory_contract.md) | [Scaling](scaling_results.md), [generated report](results/scaling/report.md) | [Summary](results/scaling/summary.json), [feature probe](results/scaling_diagnostics/feature_probe.json) |
| Template generalization | [Protocol](generalization_protocol.md) | [Results](generalization_results.md), [generated report](results/generalization/report.md) | [Address diagnostic](generalization_address_diagnostic.md), [value diagnostic](generalization_value_diagnostic.md), [fact-centric training](factcentric_training.md) |
| Context-aware distillation | [Protocol](context_distillation_protocol.md), [related work](context_distillation_related_work.md) | [Results](context_distillation_results.md), [generated report](results/context_distillation/report.md) | [Summary](results/context_distillation/summary.json), [diagnostics](results/context_distillation/diagnostics.json) |
| Counterfactual distillation | [Protocol](counterfactual_protocol.md) | [Results](counterfactual_results.md), [generated report](results/counterfactual/report.md) | [Summary](results/counterfactual/summary.json) |
| Four-step interface | [Protocol](interface_protocol.md) | [Results](interface_results.md), [generated report](results/interface/report.md) | [Summary](results/interface/summary.json), [answer errors](results/interface/answer_errors.json), [bank audit](results/interface/bank_audit.json) |
| Corpus cold start | [Protocol](coldstart_protocol.md), [literature](coldstart_literature.md) | [Results](coldstart_results.md) | [Key facts](results/coldstart/key_facts.json), [audit](results/coldstart/independent_audit.json), [literature evidence](results/coldstart/literature.json) |
| Small-data reconstruction | [Protocol](reconstruction_protocol.md) | [Results](reconstruction_results.md) | [Key facts](results/reconstruction/key_facts.json), [registration](results/reconstruction/registration.json), [audit](results/reconstruction/independent_audit.json) |
| QKV grouping and binding | [Protocol](qkv_protocol.md), [literature](qkv_literature.md) | [Results](qkv_results.md) | [Key facts](results/qkv/key_facts.json), [registration](results/qkv/registration.json), [audit](results/qkv/independent_audit.json) |
| Whole-block low-rank reader | [Protocol](block_protocol.md), [literature](block_literature.md) | [Results](block_results.md) | [Key facts](results/block/key_facts.json), [registration](results/block/registration.json), [full-query audit](results/block/independent_audit.json) |

## Interpretation and supporting material

- [TTT training-scale review](ttt_scaling_review.md): what large-scale training evidence does and does not imply for this prototype.
- [Hash-memory and initialization review](hash_initialization_review.md): source-backed initialization and memory-mechanism distinctions.
- [Latest operator diagnostics](results/block/operator_diagnostics.json), [matched Q/K check](results/block/qk_matched.json), [examples](results/block/examples.json), and [training convergence](results/block/training_convergence.json).
- [Tested environment](tested_environment.json), [English localization manifest](localization_manifest.json), and [handoff validation](handoff_validation.json).

Stage reports describe their own data, budgets, selection rules, and failure conditions. Later experiments supersede the research direction, but do not retroactively replace historical measurements. Detailed public JSON may retain legacy field names and original machine paths for traceability; explanatory text is in English.
