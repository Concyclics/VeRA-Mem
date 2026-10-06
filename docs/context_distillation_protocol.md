# From a context-conditioned teacher to a VDB–VeRA student: training protocol

This protocol was fixed before the model runs in this round. The goal is to learn a general observation writer and sparse reader so that input-generated VeRA parameter vectors reproduce the behavior of the original model when it sees the relevant text context. New facts encountered online must still require only vector writes, with no shared-weight updates. We first use the existing mechanism task for exploratory experiments; development-set improvements are not confirmatory evidence of semantic generalization.

## Literature basis and methodological limits

| Original work | Support for this design | Difference from this project |
| --- | --- | --- |
| [Askell et al., 2021](https://arxiv.org/abs/2112.00861) | Early context distillation trains a model without a prompt context to match the distribution of a model that receives it. | Knowledge is transferred into model weights; the appendix also reports that some formatting differences remain. |
| [Agarwal et al., GKD, ICLR 2024](https://arxiv.org/abs/2306.13649) | Teacher-distribution supervision on the student's own prefixes; trajectory source and KL direction are separate factors. | It does not study our writable VDB and VeRA interface; OPD is not synonymous with reverse KL. |
| [Ye et al., OPCD, 2026](https://arxiv.org/abs/2602.12275) | The teacher sees context and the student does not; reverse KL is evaluated on the same student prefix. The work uses Qwen3-4B-Instruct-2507. | It does not validate hidden-state distillation, sparse retrieval, or VeRA. Its experiments are not existing results for this architecture. |
| [Romero et al., FitNets, ICLR 2015](https://arxiv.org/abs/1412.6550) | Intermediate teacher representations serve as auxiliary training targets. | The evidence concerns image networks. Overly strong representation constraints can hurt performance, and do not guarantee that LLM states under different contexts should match exactly. |
| [Mu et al., Gisting, NeurIPS 2023](https://arxiv.org/abs/2304.08467) | A general compressor encodes new context into a compact representation in one pass, without retraining for each context. | Gist tokens are attention prefixes, rather than sparsely retrieved parameter vectors. |
| [Ge et al., ICAE, ICLR 2024](https://arxiv.org/abs/2307.06945) | Jointly learning context encoding and compatibility with a frozen language-model reader. | It uses memory slots and reconstruction/continuation training, rather than this round's OPD objective. |
| [Chari et al., KV-Distill, 2025](https://arxiv.org/abs/2503.10337) | Outputs of a frozen full-context model supervise outputs obtained from a compressed representation. | KV refers to the attention cache, not a VDB; the work does not validate VeRA's representational capacity or addressing. |

A context-aware teacher and hidden hints are therefore not new concepts. What we need to establish is whether they can train a parameter-memory interface that supports continual writes and sparse retrieval driven by layer inputs. Any claim of research novelty additionally requires a closer prior-art review and evidence on realistic tasks.

## Data flow and supervision positions

```text
Training observation c ──frozen backbone features── Wk/Wv ──temporary differentiable memory bank
                                                                       │
Question x + student prefix y<t ──frozen backbone── Wq→top-k→value→VeRA ── pS, hS
Original text c + same question x + same prefix y<t ──frozen original backbone── pT, hT
```

The teacher disables all VeRA increments, stops gradients, and reads the full observation for the corresponding training fact. The student prompt contains only the question, without concatenating that observation; the writer places the new observation in the vector bank. Teacher and student share the same frozen model and tokenizer, but run separate forwards, avoiding unnecessary vocabulary alignment. Teacher context is additional training information. Neither the teacher nor the correct fact ID is supplied to the student at evaluation.

For each output position, compare the respective `prompt_length - 1 + t` positions using exactly the same continuation token IDs. Do not align absolute sequence indices, or decode and re-tokenize the continuation. The initial hidden target is the state **after the final RMSNorm and before lm_head**, downstream of the layer-20 VeRA injection. We do not constrain the 9728-dimensional pre-injection input, which this single-layer adapter cannot alter. Only answer-prediction positions are constrained; contexts of different lengths are not forcibly aligned position by position.

The hidden-state loss is `1 - cosine(hS, stop_gradient(hT))`, weighted by 0.1, without an additional learned projection. It measures representational proximity and cannot replace memory accuracy. Context length, positional encoding, and expression format may all affect this target.

## Fixed five-arm design

| Method | Prefix at prediction positions | Main output loss | Hidden auxiliary |
| --- | --- | --- | --- |
| ce | Correct-answer prefix | Answer CE | None |
| off_kd | Correct-answer prefix | Reverse KL(S‖T) | None |
| off_kd_hidden | Correct-answer prefix | Same as above | 0.1×cosine distance |
| on_kd | Prefix sampled by the current student | Reverse KL(S‖T) | None |
| on_kd_hidden | Prefix sampled by the current student | Same as above | 0.1×cosine distance |

All four KD arms use full-vocabulary KL at temperature 1; changing KL direction is not mixed into the on-policy comparison. We use a GKD-style objective recomputed on fixed sampled prefixes, without differentiating through discrete sampling. We do not claim to implement a full policy-gradient estimator that includes gradients through the trajectory distribution. Student sampling uses temperature 1 and at most 4 new tokens, retaining actual EOS tokens and adding no artificial EOS on truncation. Unfinished examples are processed in token-wise batches; after right padding, sampling uses each example's last valid position without a KV cache. This has the same conditional sampling distribution as processing examples individually, but consumes random numbers in a different order. Off-policy training uses answer tokens followed by EOS. Losses are averaged over tokens within each example, then over examples.

All arms also use:

```text
L = L_output + 0.2 L_all_style_address + 0.2 L_actual_token_address
    + [hidden arm] 0.1 L_hidden
    + [every fourth update] 0.25 L_canonical_replay_CE
```

Each episode contains target facts and other distinct facts. All 8 query styles × 4 support styles participate in fact-identity CE. Actual prediction positions additionally use address CE over all candidate keys, addressing the lack of direct output-loss gradients to unselected keys when hard top-k misses the target. After an on-policy prefix becomes incorrect, the current question's fact remains the address label. This is an explicit training choice, not an ability to retract an already emitted incorrect answer.

The online CPU VDB path is a detached inference interface. Sampling retains no gradients. Training recomputation uses a temporary GPU bank with equivalent top-k selection; current encoders regenerate keys and values with their computation graphs intact. Adding KL directly to the detached online path would not train Q/K/value.

## Fixed budget and data scope

- Model: Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; layer-20 `mlp.down_proj`, rank/key=64, top-k=4.
- Common initialization: the final checkpoint from `factcentric_probe_20261006/anchored_all_view.pt`. This was a predetermined initialization intended to preserve compatibility with the existing value coordinates, not a choice based on this round's development results. Its uncontinued state is also evaluated.
- Use the existing cache of 4096 training entities, with 8 questions and 4 observation expressions each. There are 64 development entities; access is restricted to the train/dev branches. The old confirmation set is not read to choose methods or checkpoints. Development formats have already been analyzed in previous experiments and are not a blind test in this round.
- Each arm receives 256 updates with batch size 8, or 2048 main target-fact exposures. Every 4 steps, canonical replay of 8 facts adds 512 replay exposures. Seed 42 and the fact/view sampling sequence are shared.
- Learning rates are 1e-5 for Wq/Wk, 1e-4 for Wv, and 0.005 for shared b; gradient norm is clipped to 1 per group. The backbone, A/B, and center buffers remain frozen. Use the final step; do not select steps, loss coefficients, or arms on development results.
- Equal update and fact budgets do not imply equal token counts or FLOPs. OPD adds sampling, the teacher adds context forwards, and generation lengths differ. Record actual tokens, teacher/student forward costs, and elapsed time separately.
- Smoke tests check interfaces and gradients and are stored separately from formal results. Their scores are not used to select the recipe.

## Evaluation and stopping rules

First evaluate the context-conditioned teacher in the four original/new question × original/new observation quadrants, followed by the common initialization. Evaluate each final student in the same quadrants with real retrieval, forced correct value, shuffled values, and empty memory. Preserve per-example predictions, R@1/R@4, NLL, EM, and VDB/weight hashes. Assign new observation styles by `index % 2` and new question styles by `(index // 2) % 2`: each of the four new question–observation combinations contains 16 of the 64 entities, avoiding complete correlation between question style and the correct record's style. Do not treat different expressions as independent entities. Compare final-hidden cosine and reverse KL on a common correct-answer prefix so that differing sampled states do not make arms incomparable.

Evaluation only writes vectors generated from observations; shared weights stay frozen. Forced correct values are diagnostic, not a mathematical performance upper bound, because their distribution differs from top-k mixtures. Inspect both real−shuffled and real−empty. Greater similarity to teacher hidden states without fact-sensitive generation improvements does not establish effective memory.

This first round is a single-seed, short-budget development experiment with a 16-word answer vocabulary. Correct answers contain only 1–2 tokens, so trajectory-specific OPD effects may mainly concern the second subtoken or EOS. Results cannot be extrapolated to long-horizon reasoning. Complete and preserve the predetermined matrix regardless of outcome, rather than repeatedly increasing the budget on the same development set to choose a method.

## Subsequent validation, excluded from the first round

1. Use independent confirmation entities and structural expression families; run multiple seeds, match token budgets, and report computation. Previously analyzed XML/CSV/dialogue sets remain historical comparisons only.
2. Add genuinely multi-token values or multi-field answers. Do not manufacture an OPD advantage with irrelevant verbosity. Report first-token, complete-fact, and multi-field accuracy separately.
3. Test new-fact writes, same-entity updates, distractor records, and mixed-format banks. Measure retention after writing and whether answers change when the correct memory is replaced.
4. Independently compare post-injection block hidden states, final hidden states, context-effect differences, and hidden-loss weights. Difference vectors may still contain positional and formatting effects and cannot simply be labeled pure semantic vectors.
5. If a reliable teacher still leaves addressing or readout failures, separately study stronger writers, nonlinear Q/K, multilayer VeRA, and capacity changes. Do not attribute architectural gains to distillation.
