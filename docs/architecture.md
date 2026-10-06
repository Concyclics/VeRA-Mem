# Architecture and memory contract

The project learns a reusable writer, address function, and low-rank reader offline. At deployment, memory growth is a forward-pass encoding and record update operation. This document describes the latest implemented block path and distinguishes historical variants.

## State and dimensions

The pinned Qwen3-4B-Instruct-2507 layer-20 `mlp.down_proj` takes a 9,728-dimensional input and produces 2,560 outputs. Layer numbering is zero-based. The current rank and key dimensions are both 64.

| Object | Shape or scope | Offline learning | Online behavior |
| --- | --- | --- | --- |
| Backbone | Pinned Qwen model | Frozen | Frozen |
| A | 64 x 9,728 | Frozen random projection | Frozen |
| B | 2,560 x 64 | Trainable in the latest block study | Frozen |
| Query/key/value projections | Shared learned interfaces | Trained on allowed episodes | Frozen |
| Slot-position vectors | Three key offsets | Learned | Frozen |
| b | 2,560 output scales | Learned | Frozen |
| P_in, input bias c, P_out | Two 64 x 64 maps and one 64-vector | Learned in outer-product readers | Frozen |
| Per-fact K/V | Three 64-dimensional keys and three values | Differentiable generated episode tensors | Detached FP32 CPU records; explicitly writable |

Each fact stores 1,536 bytes of pure FP32 K/V; a 16-fact bank stores 24 KiB, excluding IDs, timestamps, and container overhead. The block's generated 64 x 64 operator is not persistently stored per fact. Both outer readers add 8,256 shared parameters and have 2,042,624 total trainable parameters; diagonal has 2,034,368. Equal parameters and storage do not imply equal FLOPs.

## Writing a fact

1. Observe the complete support text. The current microtask provides known or locatable payload word spans.
2. Encode the actual entity-plus-payload text with the frozen backbone and the memory residual disabled.
3. Pool features within each of the three word spans. Apply the shared K/V writer and key-position offsets.
4. Append or atomically replace the fact's three records in the CPU store. Record the writer version and memory event.
5. Start a new generation request against the committed bank snapshot.

This is supervised observation writing: the support can legitimately contain the later answer. A separate hidden gold-answer field is not an online input. The implementation does not yet discover arbitrary fact boundaries in unstructured documents. Rebinding training uses re-encoded contextual features for the actual new entity/payload string, rather than relabeling stale hidden states.

## Addressing and sparse group retrieval

At each eligible token position, form a query from the actual adapted-layer input `x_t`, using the learned query projection and recorded normalization. Search all CPU keys by exact cosine similarity. A fact's address score is `logsumexp_s(cos(q, k_s) / 0.05)` over its three keys. Choose the top-scoring fact and return **all three value rows**.

The storage/search path is exact CPU search in these small experiments; selected content is sparse, but search cost is not a scalable ANN result. The latest controlled evaluator activates the memory branch from the final prompt token through generated tokens; it does not broadcast one initial query over the answer. Each generated token has a saved actual query and route, including EOS. Earlier backends have their own prefill contracts, so full-prompt retrieval should not be assumed for every experiment.

In the block study, every reader uses uniform within-group weights of 1/3. Q/K receive explicit dense address-loss gradients; the discrete group choice plus uniform weights provides no answer-CE gradient to Q/K. Offline selected values and the reader remain differentiable. Online search and storage are detached.

## Three matched readers

Let `z = A x`, `m = mean_s(v_s)`, `u(v) = normalize(P_in v + c)`, and `w(v) = P_out v`. The residual added to the original frozen projection is:

```text
Delta h = b * B [M(V) z]

Diagonal:      M(V) = diag(m)
Pooled outer:  M(V) = diag(m) + w(m) u(m)^T
Block outer:   M(V) = diag(m) + mean_s [w(v_s) u(v_s)^T]
```

Multiplication by `b` is elementwise. `u` uses L2 normalization with epsilon 1e-6. Both outer maps start from identity weights; the input bias starts from an independently seeded Gaussian vector normalized to RMS 1. The block implementation can apply the factors without storing a dense per-fact matrix.

The added outer term has rank at most one for pooling and at most three for the complete block. **The total diagonal-plus-outer operator can still have rank 64**, and all readers use the same rank-64 A/B bottleneck. These experiments do not increase the width of B for the block arm.

Moving a shared linear B projection across an otherwise identical weighted sum is algebraically equivalent. The implemented change instead generates normalized input factors per row and constructs outer products before averaging. This changes both nonlinear processing order and possible outer rank; the current comparison does not uniquely attribute the gain to rank.

## Relationship to attention and KV caches

The analogy is that observed inputs generate K/V, and current token features generate Q. Persistent records can be reused without retraining at each write. The implementation differs from native Transformer attention in several respects:

- Its Q/K/V belong to an auxiliary VeRA memory interface, not the backbone's native attention heads.
- It retrieves one fact group, not multiple arbitrary chunks or the entire context sequence.
- The complete V block constructs an operator on `A x`; it is not appended as native attention-cache tokens.
- Its current block readout is invariant to row permutations. Contextual row features may encode word order, but the reader does not explicitly decode a slot sequence.
- Changing the memory bank between requests invalidates assumptions behind reusing old request activations. The bank is fixed during a generation request; retrieval changes per token.

A standard attention head also ultimately outputs a vector. Therefore, the research distinction is the content-dependent operator and retained slot interactions, not whether the final layer output is a vector.

## Implemented progression

| Module | Main distinction |
| --- | --- |
| `VectorVeRA` | Original support-derived values, sparse weighted mixture, diagonal VeRA modulation |
| `StableVectorVeRA` | Training-set centering and RMS normalization for more stable addressing/values |
| `DictionaryVeRA` | Additional offline-learnable foundation K/V bank, separate from editable facts |
| `ReconstructionVeRA` | One/three content slots; optional offline-trainable B |
| `QKVVeRA` | Shared slot writers, key position vectors, flat or grouped retrieval; additive diagnostic |
| `BlockVeRA` / `BlockBackend` | Full selected V block carried into pooled or per-row dynamic outer-product readers |

These classes are retained because immutable experiment snapshots and checkpoint contracts refer to distinct architectures. See [code map](code_map.md) for implementation locations and [block protocol](block_protocol.md) for exact training and evaluation rules.
