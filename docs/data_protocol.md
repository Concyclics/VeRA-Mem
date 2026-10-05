# Data protocol for token-level VeRA-Mem

This protocol defines what each experimental component may observe and what the initial synthetic result can establish. It describes the token-level design: queries come from the input of the adapted layer at each token, stored keys and values come from previously observed support text, and retrieved values modulate the VeRA branch. It does not treat a fixed per-request residual or a bank of independently trained LoRA adapters as equivalent implementations.

## 1. What the initial dataset tests

The primary dataset associates newly generated random entity identifiers with one of 16 common English words. The vocabulary is fixed in `data.py`; values are balanced within each split and their assignment is shuffled. Entity identifiers do not encode their answers.

This is a controlled **association-memory** task. Sharing a small answer vocabulary across training and test is deliberate: it reduces the burden of learning to generate new output strings and lets the experiment ask whether a new entity-to-value mapping can be written and subsequently addressed. It does not establish open-domain knowledge storage, arbitrary-value generalization, or conversational memory.

The current generator uses one support template, one main question template, and one alternate question template. The alternate template changes the wording of the query while preserving the exact entity string. This is a limited template-transfer diagnostic, not broad linguistic generalization. Every report must retain those qualifications.

An ideal uniform guess among the 16 known values has expected accuracy 6.25%. The actual frozen model is not necessarily uniform, so its measured accuracy remains the baseline. Successfully producing the right word is insufficient evidence of correct addressing: different entities can have the same value. Report **entity-level retrieval** as well as answer accuracy, including distractors with the same answer word.

## 2. Split contract and permitted uses

The initial target sizes are:

| Split | Facts | What may be optimized or populated | What must remain hidden |
| --- | ---: | --- | --- |
| Offline training | 128 | Learn query/key alignment, value writer, and shared VeRA read/modulation interface from support/query episodes | No development or test facts |
| Offline development | 32 | Select the checkpoint and predeclared small hyperparameter search using held-out episodes | No gradient updates on these facts; no test-driven selection |
| Online priming | 32 | Populate a fresh VDB by observing supports after the offline interface is frozen | Query labels cannot be inserted through a separate answer channel |
| Online test stream | 32 | Predict before observation; then write each newly revealed support into the VDB | No online gradient updates to the frozen primary interface |
| Unwritten controls | 16 | Read-only negative/guessing diagnostic; never populate the VDB with their support | Control support and answer must not reach the writer or query path |

For a base seed `s`, the existing generator can produce offline train, development, online stream, and priming from `s+1000`, `s+2000`, `s`, and `s+3000`, respectively. These are distinct deterministic draws, **not a proof of disjointness**. Before running, assert pairwise disjoint entity IDs and normalized support hashes across all four groups and the unwritten controls. Save the exact IDs, seeds, file hashes, code revision, model revision, and vocabulary in a manifest. Do not rely solely on the very low probability of a random-ID collision.

The primary evaluation uses four chronological blocks of eight test facts. A clean implementation measures each test block before write, observes its supports, and evaluates all observed blocks without modifying the bank. Priming recall is measured before and after the stream to show interference or dilution by new entries.

Priming has a precise meaning here: it adds **observations** to the online bank. It is not a second opportunity to train the query/key/value networks. If a baseline uses supervised gradient updates during priming or online writes, record those steps separately, give it the same observed information, and identify it as a different update mechanism.

## 3. Query and write data flow

Let `x_t` be the token input of the designated adapted module. Its dimensionality is read from that actual module, not inferred from an unrelated model hidden state. For a down-projection hook, its input is the MLP intermediate activation. The read path is conceptually:

```text
question -> frozen backbone -> x_t -> Q(x_t)
         -> sparse top-k search over the currently available VDB
         -> retrieved value mixture -> VeRA modulation for this token
```

The query path operates at the eligible token positions in prefill and subsequent generation. It does not replace all token queries with one cached last-prompt vector. If an implementation intentionally restricts retrieval positions, report the exact mask and describe that restricted variant.

The write path observes a fact through support text:

```text
observed support -> frozen backbone with memory adaptation disabled
                 -> support_x at the same designated module
                 -> K(support_x), V(support_x)
                 -> append/upsert detached key/value records
```

The answer word occurs naturally in the support, so support-derived values may contain it. That is the memory observation, not label leakage. The writer must not read the separate `answer` field or concatenate the later test question to manufacture a key. Both token-level writing and a fixed support-token selection/pooling policy are viable experimental variants; state the selection exactly. In a causal model, a support token before the answer word cannot have observed that word. If using the last content token or pooling, record the positions and exclude padding consistently.

Offline episodes teach alignment between answer-free query-token features and the **support-derived** keys of the corresponding observation. The known support/query pairing may identify positives in a contrastive training loss. It must not become an inference-time lookup from an example ID to its correct key. Use other entities, including entities sharing the same answer value, as negatives to avoid reducing addressing to a 16-way value classifier.

During online evaluation, freeze the backbone and learned read/write interface. Adding records changes the bank; it does not update the projection parameters. Record the encoder/checkpoint version with the bank. Re-encoding a stored bank with a later encoder version requires an explicit migration condition, not an invisible change during evaluation.

The VDB snapshot is fixed throughout one generation request; individual tokens may retrieve different records from that fixed snapshot. Do not write model-generated answers back during evaluation. If the bank changes between requests, recompute the prompt state for the new request rather than carrying over incompatible cached activations.

## 4. Public JSONL schema and access roles

The existing `Example` schema is retained. The following is an illustrative public synthetic line, not a claimed row from a completed run:

```json
{"id":"synthetic-N4fa73d980c216be5","question":"What is the assigned memory word for entity N4fa73d980c216be5? Reply with only the word.","answer":"river","support":"The assigned memory word for entity N4fa73d980c216be5 is river.","paraphrase":"Recall the memory word associated with N4fa73d980c216be5. Return only that word.","choices":null,"metadata":{"dataset":"synthetic-associations-v1","seed":42,"entity":"N4fa73d980c216be5","split":"stream","never_written":false}}
```

All fields may coexist on disk for reproducibility; that does not authorize passing the full object to every model component.

| Field | Query/generation input | Writer input | Offline supervisor / scorer |
| --- | --- | --- | --- |
| `question` or selected `paraphrase` | Yes, one at a time | No; support-derived keys are required in the primary design | Yes |
| `support` | No in latent-memory inference; yes only as retrieved context for an explicitly named text-RAG baseline | Yes, after its observation event | Yes |
| `answer` | No | No separate access; the observed support may naturally contain the same word | Offline answer loss and read-only evaluation only |
| `id`, `metadata.entity` | No direct addressing shortcut; the entity text already occurs in the natural query | May identify records for lifecycle/logging, not form a handcrafted semantic key | Positive pair/evidence identity, joins and scoring |
| `metadata` labels, source labels | No | No label-based routing | Audit and stratified metrics |
| `choices` | Only as the declared multiple-choice task input | Only if present in observed support | Option mapping and scoring |

A control example has the same shape but `split="control"` and `never_written=true`. Its `support` is a **withheld counterfactual observation**, not available history. The current generator still assigns it a random known-word `answer`. Therefore control exact match is an unexposed guessing diagnostic; it is **not** an unknown-answer or abstention score.

Generation must finish and save its output before the evaluator consumes `answer`. Teacher-forced NLL may then use the gold answer in a separate read-only forward pass. Later answer positions in that pass contain earlier gold tokens, so their retrieval statistics must not be presented as answer-free inference behavior. For causal addressing metrics, log retrieval from the final prompt position and actual generated positions, with the token/position mask specified. Do not average over every irrelevant punctuation/system token to produce a misleading global recall.

## 5. Core conditions that fit the first run

The following conditions use the existing synthetic facts and can be run before adding new data generators:

1. **Frozen/no memory:** establishes behavior on unseen associations.
2. **Text oracle:** exposes the legally observed correct support as context; checks whether the base model and answer parser can solve the task.
3. **Text lexical retrieval:** indexes all currently observed supports; no filtering by the known correct example ID. This simple baseline is expected to be strong when the exact entity string is shared.
4. **Primary token-level VeRA-Mem with real retrieval:** token queries come from `x_t`; all keys/values come from observed supports. Report its actual retrieval behavior.
5. **Oracle vector retrieval:** restricts retrieval to the legally observed correct support records. The scorer's evidence ID may select that support only for this explicitly named diagnostic. It never creates an unseen fact or supplies the output label.
6. **Empty bank and shuffled/wrong values:** preserve the trained interface and comparable vector dimensions. Reproduce the same data order and snapshot. For a wrong-value intervention, deliberately use a different answer value: a random shuffle over 16 values can accidentally preserve the correct word. Record both entity changes and value changes.
7. **Existing alternate question template:** evaluate without any additional write or optimization. Label the result as alternate-template performance.

The comparison separates failure modes. Text oracle failure suggests a task/format/backend issue. Vector oracle success with real-retrieval failure points to addressing. Correct vector evidence and shuffled values yielding the same performance suggests the model is not using the value content. Improvement with an empty bank may instead reflect shared-parameter adaptation or an answer prior.

At 32 facts, each extra correct answer changes accuracy by 3.125 percentage points. Report counts alongside percentages. A single seed is a feasibility/diagnostic run, not enough to establish a reliable method advantage. Do not choose thresholds or the primary metric after looking at test outcomes.

## 6. Additional task families and when to run them

These require explicitly versioned data and protocol changes. They must not be implied by the current `synthetic-associations-v1` results.

### 6.1 Conflicting updates and latest-value queries

This is a useful second short run after the ordinary write/read path works. Reuse an entity, reveal a later support with a **different** value, and query the latest value before and after that event. Keep separate `event_id`, `entity`, `attribute`, and `timestamp`. An illustrative event-schema record, **not directly loadable by the existing Example loader**, is:

```json
{"event_id":"update-002","entity":"N4fa73d980c216be5","attribute":"memory_word","timestamp":2,"support":"Update at time 2: the current memory word for entity N4fa73d980c216be5 is apple."}
```

Choose one storage policy and name it: either latest-only upsert keyed by a declared entity/attribute record identity, or append-only temporal records with a time-aware query/read policy. Plain cosine similarity plus insertion-order tie-breaking does not implement latest-value semantics. Test cases must include unchanged distractor entities and repeated updates. Report latest-value accuracy and superseded-value error rate.

History-at-time queries are a **separate later condition**: they require retained version history. They are not fairly solvable by a latest-only store, and returning an old value to an explicit old-time question is not forgetting or update failure.

### 6.2 Unknown entities and missing evidence

The current random-label controls cannot score abstention. A true unknown condition requires the answer target `UNKNOWN`, an instruction defining that output, and comparable unknown episodes during offline training/development if a learned gate is expected to abstain. The bank must contain distractors but no support for the query entity/attribute. Use both an empty bank and a populated irrelevant bank; these are different tests.

For the first run, report only empty-bank behavior, control guessing accuracy, and whether a retrieval fallback injects unrelated values. A **scored unknown task is the next stage**, with known-answer recall, unknown false-positive rate, abstention precision/recall, and any similarity threshold selected on development data. Do not add `UNKNOWN` to the test prompt alone and interpret a zero-shot failure as evidence against the memory architecture.

### 6.3 Held-out values

Entity-disjoint evaluation still shares all 16 answer words. For value generalization, split the value vocabulary before constructing facts and keep test values out of offline training, development selection, and priming. First use additional common words matched approximately for tokenizer length; then test multiword values or random codes as a harder task.

This is **after the initial addressing run**. It changes the requirement from associating known output symbols to transporting unseen content through the value/modulation path. Evaluate text oracle, vector oracle, and real retrieval separately. A drop specific to held-out values identifies an encoder/reader extrapolation problem; it does not erase a demonstrated ability to remember new entities within the trained vocabulary. Do not use a decoder restricted to the original 16 words in this condition.

### 6.4 Paraphrases, aliases and distractor structure

The built-in alternate template can be evaluated immediately. A broader follow-up should reserve whole support/query template families from offline training, vary entity location, add intervening text, and include multiple attributes for the same entity. Save template IDs and ensure development and test template families are disjoint when claiming template generalization.

Entity aliases are a distinct difficulty: if a query omits the original identifier, the alias relation must itself have been observed. Otherwise the test is unanswerable, not simply a harder paraphrase. Semantic paraphrases must preserve the same target fact and be manually checked or validated against a deterministic generator; do not let a paraphrase-generation model inspect or accidentally insert the answer into the query.

### 6.5 Larger streams and scaling

After checking value use and real retrieval, increase the number of distinct facts and distractors while fixing offline training, vocabulary, and read budget. Report entity recall@k and final answer EM versus bank size, plus write/read costs and actual bytes per stored support/token. If writing multiple vectors per support, fact count and vector count must both be shown. The initial small stream cannot demonstrate long-term retention merely because it contains two passes.

## 7. MedMCQA is a transfer diagnostic

MedMCQA changes nearly everything beyond the formal support/query distinction: sequence length, question structure, domain knowledge, answer format (A–D), value semantics, and the number of tokens carrying relevant evidence. A writer trained only on short synthetic statements mapping identifiers to 16 words has not been trained to encode medical QA supports. Its unadapted transfer failure is **not a valid rejection of the architecture**.

The appropriate staged comparison is:

1. Frozen model, text oracle and text retrieval on a small pinned MedMCQA subset.
2. Zero-shot synthetic-interface transfer, explicitly labeled as out-of-distribution diagnostics.
3. If needed, domain-aligned offline writer/read training on a separate MedMCQA train subset; development and online-memory test questions remain disjoint by ID and normalized question/options hash.
4. Compare vector oracle against real retrieval with that adapted interface before drawing conclusions about medical memory.

The loader's current source `cop` schema is zero-based (0→A through 3→D); it excludes `exp` and downloads only train and validation parquet files at a pinned revision. The support reveals the correct answer through the observed QA record. The read query contains the question/options only. The existing MedMCQA `paraphrase` changes the instruction wrapper, not the meaning/wording of the medical question. Public medical questions may have appeared in pretraining; increased repeated-question accuracy must be called learned/retained QA performance, not proof of novel-knowledge acquisition. [Dataset source](https://huggingface.co/datasets/openlifescienceai/medmcqa)

## 8. Required audit artifacts and claims

Save per split: seed, entity and event IDs, vocabulary/template IDs, source revision if external, normalized content hashes and JSONL SHA256. Save per run: offline training/dev membership, selected checkpoint criterion, online write order, observed support IDs at each snapshot, number of facts and vectors, token positions used for write/read, top-k, retrieved IDs/scores and generation outputs. The evaluator may keep labels next to predictions in result files; model-facing interfaces should receive only their permitted views.

Before accepting a run, verify:

- No entity/support overlap among offline train, development, priming, test and unwritten controls.
- Changing the hidden `answer` field while holding the query/support views fixed does not change online generated output or VDB contents; only scoring changes. This test excludes offline training, where labels legitimately supervise the interface.
- Withheld control supports are absent from every bank snapshot.
- Online evaluation changes neither projection/VeRA parameters nor memory contents; only explicit write events alter the bank.
- Stored keys are encoded from supports; a test-question embedding is not silently used as the stored key.
- Read retrieval uses actual layer-token inputs, and evidence labels are only used in named oracle/scoring paths.

The first result should answer a narrow, useful question: **after learning an interface on other entities, can a frozen token-level memory mechanism write a newly observed association, address its support from the query's layer inputs, and use the retrieved value to generate the correct known word?** Larger-vocabulary, temporal, abstention and open-domain claims require the additional conditions above.
