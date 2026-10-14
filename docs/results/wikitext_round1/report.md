# WikiText-103 first-round results

One complete training pass, seed 20261013. Article-bootstrap intervals describe this held-out sample; they do not measure training-seed robustness. Comparisons are exploratory and are not adjusted for multiplicity.

Training LM targets: **124,273,491**. Additional episodic targets: **613,400**.

## Static language modeling

| Method | Clean test PPL | PPL / frozen base |
| --- | ---: | ---: |
| base | 9.8387 | 1.0000 |
| diagonal | 8.7185 | 0.8861 |
| factor_mean_outer | 8.2674 | 0.8403 |
| block_outer | 8.2732 | 0.8409 |
| lora | 7.5185 | 0.7642 |

The train-overlapping article is excluded from primary test PPL. The machine-readable result also retains the full official split statistic.

## Frozen-weight continual memory

| Reader | Stream A EM gain vs pre-write (pp, 95% CI) | B gain vs stale A (pp, 95% CI) | Joint A/B/restore EM |
| --- | ---: | ---: | ---: |
| diagonal | +0.05 [+0.00, +0.18] | +0.00 [+0.00, +0.00] | 0.00% |
| factor_mean_outer | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | 0.00% |
| block_outer | +0.00 [-0.16, +0.14] | +0.00 [+0.00, +0.00] | 0.00% |

These are three-word anchored passage completions with two unseen query styles, not open-domain QA. All reader/writer parameters and statistics were unchanged during writes, edits, and restoration.

## Interpretation limits

A static PPL gain alone demonstrates corpus adaptation. Evidence for online memory requires paired gains after new writes and counterfactual edits, together with restoration and retention. The full JSON contains pre-write, shuffled, oracle, retrieved-text, and LoRA controls, paired operator comparisons, and all diagnostic gates. No gate caused early stopping.

This matches the WikiText corpus scale, not the parameter counts, multi-epoch budgets, backbones, or reported perplexities of Memory Decoder or MLP Memory.
