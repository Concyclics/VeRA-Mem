# Whole-block value reads and dynamic low-rank memory: literature evidence

Building on the [existing QKV literature map](qkv_literature.md), this round checks only the three most directly relevant primary sources. The conclusion is: **whole-block retrieval has precedents, and memory-generated dynamic matrices have a rigorous theoretical connection; however, moving a shared linear projection before mixing does not increase expressiveness, and no paper guarantees that this will resolve the failures on our controlled novel-combination task.** See [block_protocol.md](block_protocol.md) for the falsifiable experimental design.

## Distinguish three operations first

Standard attention with one query and one head still outputs a vector: `m=Σ_s α_s v_s`. For shared linear B and fixed current input `z=Ax`:

```
B (Σ_s α_s v_s) = Σ_s α_s B v_s
B [(Σ_s α_s v_s) ⊙ z] = Σ_s α_s B (v_s ⊙ z)
```

With identical weights and no intervening nonlinearity or slot-specific mapping, moving the sum later or B earlier merely changes an equivalent computation order. Retrieving a whole block and applying these same equations does not establish additional capacity. Every reader must eventually produce a vector for the next layer; the object to test is the **input-dependent operator**, rather than treating “the final output is a vector” as the cause of failure.

This round applies the per-slot outer product `w(v_s)u(v_s)ᵀ` to z. This permits off-diagonal coordinate mixing, unlike the original `diag(m)z`. The controls retain the common `diag(m)` and compare `w(meanV)u(meanV)ᵀ` with `mean_s[w(v_s)u(v_s)ᵀ]`, with exactly the same parameters. Because u includes normalization and a bias, this comparison changes both the order of factor generation and pooling and the rank of the outer term. Results cannot be attributed uniquely to rank or matrix storage format.

## 1. Fast Weight Programmers: content outer products can construct a dynamic matrix

**Schlag, Irie & Schmidhuber, Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021.** [Paper §§3.1–3.2](https://arxiv.org/html/2102.11174v3#S3), [§§4.1–4.2](https://arxiv.org/html/2102.11174v3#S4), [authors' code](https://github.com/ischlag/fast-weight-transformers).

The paper writes attention without softmax as `V(Kᵀq)=(VKᵀ)q`; `W=Σ v kᵀ` is a fast-weight matrix generated from the current content. Ordinary softmax attention remains a weighted sum of values. It then analyzes key interference in finite-dimensional associative memory and introduces delta writes based on the error in the previous read. Synthetic experiments use random key/value associations and repeated-key updates, but do not directly establish that a frozen LLM can generate novel three-word combinations.

The implication for this experiment is to retain each fact block, transform its content into a small dynamic operator, and apply that operator to the layer input. Finite rank, interference, and writer/readout compatibility remain limitations. We do not accumulate the entire database into one global fast-weight state: each VDB fact remains independently writable and replaceable, followed by sparse selection of a block. The paper therefore supports the mathematical design, rather than directly proving our write locality or generalization.

## 2. DeltaNet: a useful write rule is more specific than “having a matrix”

**Yang et al., Parallelizing Linear Transformers with the Delta Rule over Sequence Length, 2024.** [Paper §2.2](https://arxiv.org/html/2406.06484v3#S2.SS2), [method §3](https://arxiv.org/html/2406.06484v3#S3), [official implementation](https://github.com/fla-org/flash-linear-attention).

DeltaNet's core update can be written as `S_t=S_(t−1)+β_t(v_t−S_(t−1)k_t)k_tᵀ`: retrieve the previous value for the same key from the matrix, then write the error. The paper also parallelizes this recurrence and evaluates associative recall and large-scale language modeling. Its results do not imply that replacing a VeRA vector with a matrix in an already trained model will produce generalization.

Our VDB already replaces the target record exactly; a delta rule is not required for individual upserts. This round is **not a DeltaNet reproduction**. If many facts are later compressed into one fixed matrix, error-correcting writes, forgetting, collisions, and order sensitivity will warrant explicit comparisons. This round first compares low-rank readout, avoiding a simultaneous change to the write rule and physical storage permissions.

## 3. LongMem: block retrieval retains token structure but still ends in attention

**Wang et al., Augmenting Language Models with Long-Term Memory, 2023.** [Paper §2.3](https://arxiv.org/html/2306.07174#S2.SS3), [training setup §3.1](https://arxiv.org/html/2306.07174#S3.SS1), [code entry supplied by the paper](https://aka.ms/LongMem).

LongMem retrieves using mean chunk keys, fetches complete chunk K/V, and expands them into token-level K/V for softmax attention. Its memory output is still one weighted vector per query, gated with local attention. The configuration uses a frozen 407M backbone and a trainable 12-layer SideNet, trained on 26B tokens; each token retrieves 16 four-token chunks, totaling 64 K/V pairs.

This directly supports different granularities for indexing and returned content: sparsely retrieving whole blocks can retain neighboring information. It does not use the whole V block as a dynamic low-rank parameter matrix for VeRA. Its training scale and SideNet capacity also greatly exceed our single-layer controlled experiment. We borrow the structural rationale for whole-block reads without treating its success as the expected outcome of our experiment.

## What this round can claim

Only if block_outer consistently exceeds pooled_outer with the same regime, seed, data, target exposure, shared parameter count, K/V bytes, and diagonal path—and actual content updates outperform shuffle/empty—would the evidence support per-slot operator construction over pooling first for this task. Cost must reflect the actual operator computation; equal parameters and storage do not mean equal FLOPs.

If improvement appears only against diagonal, additional parameters and more complex factor generation remain explanations. Failure on novel combinations would not invalidate all fast weights or attention. High fact recall with a wrong answer cannot isolate value encoding: subsequent decoding may drift, and wrong prefixes and routing changes may affect one another. Located three-word spans, a finite vocabulary, single-word substitutions, and a finite optimization budget limit extrapolation. This round establishes neither automatic fact discovery from unrestricted long documents, cross-domain continual learning, nor a priority claim.
