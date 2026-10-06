# Context distillation and retrievable parameter memory: closely related work

Reviewed on 2026-10-06. This document adds research context without changing the fixed [five-arm exploratory protocol](context_distillation_protocol.md). Method descriptions come from the original papers below. The comparisons with our design and the proposed validation steps are our analysis, not conclusions those authors reached about this project.

Prior work already covers context-conditioned teachers, on-policy distribution supervision, hidden-state alignment, one-pass adapter generation, and retrievable parameter-memory banks. This project therefore cannot claim those individual concepts, or an insufficiently checked combination of them, as first-of-their-kind contributions.

## Six particularly relevant papers

### 1. OPCD: direct support for context-aware on-policy distillation

[Ye et al., *On-Policy Context Distillation for Language Models*, 2026](https://arxiv.org/html/2602.12275), §3 and §3.1.

- **Teacher context:** The teacher receives additional context; the student does not receive the original text. Next-token distributions are compared on the same student-generated prefix. A frozen teacher initialized from the same backbone can be used.
- **On-policy:** Yes. The student samples first and is then trained with `KL(student || teacher)`. The paper's implementation also approximates KL using the student's top-k tokens.
- **Hidden supervision:** The main method does not add hidden-state matching.
- **Writing and reading:** Training transfers context-induced behavior into student parameters; inference directly uses the updated model.
- **Difference from this design:** We train a reusable writer/reader, while new online facts only generate and write VDB vectors. Each token retrieves VeRA values from its layer input. OPCD does not validate that sparse read/write chain. Our current experiment uses full-vocabulary KL and is not an exact reproduction of its implementation.

### 2. GKD: separating rollout source from the distillation objective

[Agarwal et al., *On-policy Distillation of Language Models: Learning from Self-Generated Mistakes*, ICLR 2024](https://arxiv.org/html/2306.13649), §3 and §3.1.

- **Teacher context:** The framework does not require the teacher to possess factual context missing from the student; its core is teacher-distribution supervision on the same sequence prefix.
- **On-policy:** Optional. GKD explicitly controls the mixture of student trajectories and fixed data, without backpropagating through discrete sampling.
- **Hidden supervision:** The main objective compares token distributions, rather than regressing hidden states.
- **Writing and reading:** Student parameters are updated; no separate memory-entry and retrieval interface is designed.
- **Difference from this design:** GKD motivates training controls rather than the parameter-memory architecture. It selects trajectory source separately from forward KL, reverse KL, and JSD: **on-policy is not synonymous with reverse KL**. Our off/on arms hold reverse KL fixed to test the additional effect of sampled prefixes.

### 3. SADA: the closest precedent for context-conditioned states guiding dynamic adapters

[Gao et al., *SADA: Bridging In-Context Learning and Fine-Tuning via State-Aligned Distillation Adapters*, ACL 2026](https://aclanthology.org/2026.acl-long.1046/), [original PDF](https://aclanthology.org/2026.acl-long.1046.pdf), §3.2 and §3.3.

- **Teacher context:** The teacher uses the full history; the student uses truncated context and dynamic parameter updates.
- **On-policy:** The described training operates on pretraining sequences or task targets, without a data-collection mechanism centered on student rollouts.
- **Hidden supervision:** Yes. Stage I uses layer-wise hidden MSE and forward KL; Stage II uses hidden MSE and the target-token LM loss.
- **Writing and reading:** Attention outputs from evicted history are compressed in chunks and incorporated through recurrent state updates. Meta-LoRA injects static A and memory-generated dynamic B into a frozen model.
- **Difference from this design:** SADA already covers state and behavioral distillation for dynamic adapters. Its central memory is an evolving state; it does not propose our independent VDB entries with layer-input-driven, token-wise sparse value retrieval. Our first round aligns only the final RMSNorm hidden state, rather than its layer-wise MSE.

### 4. Doc-to-LoRA: the closest precedent for writing a new document in one forward pass

[Charakorn et al., *Doc-to-LoRA: Learning to Instantly Internalize Contexts*, 2026](https://arxiv.org/html/2602.15902), §3, §5, and Appendix B.1.

- **Teacher context:** A frozen original model reads the document and question, supplying distribution targets for an adapter-equipped student that does not receive the original text.
- **On-policy:** Main training uses pregenerated teacher answers and saved logits with forward KL, rather than online student rollouts.
- **Hidden supervision:** Frozen-model layer activations are hypernetwork inputs. This is not itself a hidden-state distillation loss; the main objective does not match hidden states layer by layer.
- **Writing and reading:** A Perceiver hypernetwork maps context activations to LoRA A/B for each layer's MLP `down_proj`. A new document produces an adapter in one forward pass, after which the original text need not be repeated. Long documents combine chunk-generated adapters through rank concatenation.
- **Difference from this design:** There is already a direct precedent for new context without separate gradient training. We aim to store small VeRA vectors under shared fixed A/B and select memories dynamically inside inference. D2L's main method generates a document-level adapter without this kind of token-wise VDB reading.

### 5. Latent Memory Management: existing parameter-memory banks and retrieval

[Zheng et al., *Context Distillation as Latent Memory Management*, 2026](https://arxiv.org/html/2605.28889), §3 and Appendices C and D.

- **Teacher context:** A context-conditioned backbone supervises a corresponding LoRA student for each document.
- **On-policy:** Document writing uses synthetic QA and independent adapter distillation. The paper also reports reverse-KL, top-k-logit, EMA, and other variants. Those objective names alone do not establish that all variants use the same student-rollout protocol; this factor is not independently controlled as in our five-arm design.
- **Hidden supervision:** Hidden states serve as candidate-adapter routing features, rather than a teacher–student hidden-alignment loss.
- **Writing and reading:** Each document receives separate gradient-based LoRA training. External text embeddings retrieve top-k candidates by cosine similarity; first-token hidden states and entropy then select an adapter and gate whether to enable it.
- **Difference from this design:** We cannot claim to be the first to retrieve parameter memory. The unverified distinctions are a general writer that adds new entries without gradients, internal layer-input queries at each token, and sparse mixtures of several small VeRA values, rather than selecting one complete document adapter after external retrieval.

### 6. Cartridges: distilling context into reusable compact memory

[Eyuboglu et al., *Cartridges: Lightweight and general-purpose long context representations via self-study*, 2025](https://arxiv.org/html/2506.06266), §4.1, §4.2, §5.4, and §6.

- **Teacher context:** The teacher reads a corpus or subcorpus; the student uses a trainable short KV cache.
- **On-policy:** The main method distills synthetic self-study conversations, rather than current-student online rollouts.
- **Hidden supervision:** Its core is next-token distribution matching with forward KL, without the hidden-alignment term used here.
- **Writing and reading:** A KV cache is separately optimized for each corpus and loaded for inference. Independently trained cartridges can be concatenated. Subsequent queries amortize training cost.
- **Difference from this design:** A cartridge is an attention KV prefix and still requires corpus-specific training. It does not provide a general writer that adds a new fact in one pass, or internal sparse retrieval of VeRA parameter vectors.

## Positioning this project

The current research question can be stated as:

> Can context-conditioned behavioral and state distillation learn a general writer/reader interface that encodes a new observation once into VeRA parameter vectors under a shared fixed low-rank basis, retrieves them sparsely from internal layer inputs at each token, and uses new facts without repeating the original observation or updating shared weights online?

This is a **hypothesis about composition and efficiency**, not established novelty or performance. SADA makes the hidden-distillation precedent more direct; D2L covers amortized context-to-adapter generation; Latent Memory Management covers retrievable parameter memory; and OPCD covers an on-policy teacher with additional context. Our five-arm experiment only tests whether distillation improves an existing VeRA–VDB interface. It cannot establish superiority over those complete methods.

## Evidence still needed to distinguish the approach

1. **Write cost and storage:** Verify that new facts need only a forward write. Report bytes per memory, write latency, indexing cost, and shared writer/basis costs. Differences from document-specific LoRA training cannot be described solely through post-training inference latency.
2. **Read granularity:** Compare external-embedding selection of one adapter, fixed question-level retrieval, and dynamic retrieval from the current layer input at every token. Test whether internal addressing provides an additional benefit.
3. **Capacity and budgets:** Single-layer rank-64 VeRA and multilayer generated LoRA have different capacities. Specify parameter, memory, forward-compute, and training-data budgets; do not attribute capacity differences to OPD or hidden loss.
4. **Causal use of memory:** Test writes, updates, deletions, multi-fact composition, and distractors; compare real, shuffled, empty, and forced correct values. More similar hidden states or lower training loss alone do not establish effective memory.
5. **Generalization and interference:** Test new entities, structural expressions, genuine multi-token values, irrelevant queries, and multiple seeds. The current synthetic short-answer development results support only limited mechanism-level conclusions.

These papers justify experimentation while showing that the goal requires more than swapping a distillation loss. If the direction is effective, the key evidence will be behavioral gains, costs, and update capabilities of compact writable memory under actual sparse retrieval.
