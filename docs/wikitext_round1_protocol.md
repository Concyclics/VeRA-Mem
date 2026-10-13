# WikiText-103 first round: full-corpus shared Q/K/V learning

Protocol date: 2026-10-13. This experiment follows the [scale review](paper_scale_alignment.md). Its status and measured results are maintained separately in the [execution record](wikitext_round1_execution.md). A successful engineering pilot does not constitute a completed corpus experiment.

## Scope and claim boundary

Run one complete pass of the official WikiText-103 training text with **Qwen3-4B-Instruct-2507 frozen**, comparing three sparse-memory operators and a parameter-matched LoRA control. The first round uses seed `20261013`. A 10M-token checkpoint is evaluated for diagnosis, after which all arms continue regardless of quality. Three-seed replication and additional corpora remain later work; this first round cannot establish robustness across training seeds.

This aligns the corpus scale with [Memory Decoder](https://arxiv.org/html/2508.09874v2) and [MLP Memory](https://arxiv.org/html/2508.01832v3), not their architectures, optimization budgets, or published perplexities. The original papers use different backbones, tokenizers, substantially larger memory predictors, and multiple training epochs. Our sparse teacher represents eight causal endpoints per 128-token block rather than a dense entry for every corpus token. No head-to-head superiority claim is justified by this study.

## Data and leakage controls

- Dataset: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-103-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. All four official parquet shards are used.
- Backbone: `Qwen/Qwen3-4B-Instruct-2507`, intended revision `cdbee75f17c01a7cc42f958dc650907174af0554`. The actual local model files are recorded in the private provenance artifacts.
- Preserve text rows, headings, and empty lines. A document boundary requires a single-equals outer heading and a blank row on each side. Internal equals signs, such as those in an album title, are retained. These are conservatively reconstructed document units, not a claim that every original Wikipedia boundary can be recovered from plain text.
- The final preparation contains **124,245,019 training text tokens** and **124,273,491 causal targets**, including one EOS per document. It reconstructs 28,472 training units, 60 validation units, and 60 test units. Every training target is scored once during the corpus pass; extra episodic targets are counted separately.
- Keep all official training text, including 241 repeated article instances. Group duplicates by case-folded title or stripped-text identity, and exclude the entire group from a training query's memory candidates.
- One test article, *The Hustler (film)*, overlaps training. Report the full official test result as a secondary contaminated-split statistic and exclude this article from primary test PPL and continual confirmation.
- Validation/test text never initializes the static training memory or normalization statistics. Online confirmation writes only the supports already observed in the current stream minibatch.

The preparation audit caught equation/statistics rows that resemble headings. The initial cache is retained as an auditable preprocessing attempt. Corrected features reuse a previous row only after a SHA-256 lookup **and exact equality of all observed token IDs**; unmatched chunks are re-encoded. Statistics are recomputed over the corrected training cache. No learning result from the incorrect segmentation is used.

## Versioned memory interface

The new [corpus modules](../src/vera_mem/corpus_reader.py) are separate from the historical three-slot implementation.

| Component | Fixed setting |
| --- | --- |
| Injection site | Input/output of layer 20's MLP down projection; 9,728 to 2,560 dimensions |
| Observation | At most 128 observed tokens, encoded with the frozen backbone prefix |
| Writer cache | Eight spans of at most 16 tokens; a fixed seeded projection to 512 dimensions |
| Coarse index | 128-dimensional centered, normalized block vectors; Faiss GPU IVF-Flat with 4,096 centroids |
| Candidate read | 32 candidate blocks, learned fine Q/K reranking, four complete selected blocks |
| Fine representation | Q/K width 64, V width 64, common VeRA rank 64 |
| Trainable maps | Two-layer shared Q/K/V maps, coarse query projection, common output projection/gain, and declared outer factors |
| Inference payload | Eight low-rank K/V rows per block; no answer labels and no 512-dimensional writer cache required |

Queries originate from the **actual VeRA-layer input**. The index coordinates and training-derived center stay fixed. A trainable coarse query projection addresses this space; current fine K/V weights re-encode candidate features during offline training. Thus, changing trainable K/V weights does not silently stale an index built from those weights. At inference, the writer is frozen and new observations directly produce low-rank K/V payloads. Logical upserts append a new physical version and immediately mask the previous version. The index and payload commit before any query is admitted.

The index is qualified against exact inner-product search using 256 actual query activations from eight training windows, with identical article exclusion. GPU IVF-Flat uses 4,096 spherical centroids, trained for 25 iterations on a fixed sample of up to 262,144 training block vectors. Test `nprobe` values 8, 16, 32, 64, 128, 256, 512, and 1,024. Select the smallest reaching mean recall@32 of 95%; otherwise use the maximum and explicitly report the recall shortfall. This is an engineering choice based on training data, not test-quality tuning. Record build time, recall, query throughput, memory, and serialized bytes. See [Faiss index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

Before formal training, CPU HNSW with 32 graph neighbors measured recall@32 of 43.8%, 56.6%, 68.8%, and 80.3% at `efSearch` 64/128/256/512. At 512 it processed about 2,078 queries/s; index search dominated the engineering pilot. This prompted the GPU IVF implementation, using the [official GPU package](https://github.com/facebookresearch/faiss) in an isolated experiment dependency directory. The same pilot caught a Transformers chat-template return-type incompatibility, fixed before formal training. These are retained engineering attempts, not quality results.

For selected V rows and shared soft weights, let `m = sum_s a_s v_s`, `z = A x`, `u_s = normalize(U v_s + b)`, and `w_s = W v_s`. Compare:

1. **Diagonal:** `diag(m) z`.
2. **Factor-mean outer:** `diag(m) z + (sum_s a_s w_s)(sum_s a_s u_s)^T z`.
3. **Block outer:** `diag(m) z + sum_s a_s w_s u_s^T z`.

The outer arms have identical factor parameters, initialized identically. Factors are computed before pooling in both arms. The diagonal arm has 8,256 fewer trainable parameters; exact counts are logged. The LoRA rank is rounded to match the block arm's parameter count. All arms receive the same corpus windows and episode schedule. They share the candidate algorithm and initialization; learned queries can subsequently produce different candidate IDs.

This changes several mechanisms relative to the small-data experiment: eight slots, multiple blocks, content-dependent row weights, an indexed learned route, compact cached features, and corpus/episodic training. It is not a pure data-size ablation.

## Training and accounting

Corpus windows have at most 1,024 input positions, stay within a document, and cover all next-token targets exactly once. Shuffle the complete window table with the fixed seed. Train the three readers and LoRA using the same batches. Reuse the frozen prefix for identical inputs; reconstruct `residual + (down_output + delta)` in the original BF16 addition order. Tests and a full-model qualification compare this with an ordinary down-projection hook and check incremental KV caches.

Use AdamW, peak learning rate `1e-4`, two million LM tokens of warmup, cosine decay to 10% of peak, weight decay `0.01`, and gradient norm limit `1`. The backbone stays in BF16/evaluation mode; adapters/losses are FP32 with TF32 matrix multiplication enabled. The engineering pilot fixes a feasible batch size before formal registration. Checkpoint optimizer state, RNG state, data cursor, separate token counters, and source/input hashes. Save a checkpoint at approximately every five million LM targets and at completion.

For sparse readers, the corpus objective is next-token CE plus `0.1` sparse retrieval-target CE, `0.1` fine-address CE, and `0.1` coarse-route CE. Each observed span supplies a causal key immediately before its last observed token; that token supplies a sparse teacher label. The frozen-feature teacher compares the current query with shortlisted causal keys and retains 16 slot probabilities. Duplicate vocabulary IDs are aggregated implicitly. Sparse CE has the same student gradient as KL plus a constant teacher entropy; logs must not label its numerical value as KL.

Every eight corpus updates, add `8 * corpus_batch_size` natural write/read episodes from 16,384 fixed training supports. Each query supplies an eight-word anchor and requests the next three words. Use two training phrasings. The loss is answer CE plus `0.5` contextual teacher CE and the two `0.1` address terms. A frozen Qwen teacher sees the observed support; the student does not. The teacher distribution stores its top 32 probabilities and an explicit remaining-mass bucket. This is **offline teacher-forced contextual distillation**, not on-policy distillation. Force the observed positive into training candidates for three out of four episode updates; use natural ANN for the fourth. Confirmation never forces a positive except in its separately labeled oracle controls.

LoRA receives the same corpus CE and contextual episode supervision. It has no VDB and therefore no address objectives. Its online baseline keeps weights frozen; this does not measure a per-observation LoRA retraining baseline. Empty-memory VeRA is exactly the frozen base by zero-injection construction. Shuffled-memory validation measures harm from irrelevant values.

## Two separate evaluations

**Static LM:** the memory contains training observations only. Evaluate all validation/test targets with up to 1,024 input positions and a stride of 512; initial/tail windows can score fewer targets. Each target, including document EOS, appears exactly once. Report NLL, PPL, article-level sums, frozen-base change, and LoRA comparison. Report value-shuffled validation PPL separately. Do not ingest held-out text to improve static PPL.

**Continual memory:** freeze all reader/writer weights and statistics; hash them before and after the entire evaluation. Confirmation has 1,000 deterministic held-out units from 58 non-overlapping test articles, with two query phrasings absent from training. The development checkpoint uses 64 validation units. This is anchored natural-passage completion, not open-domain QA or evidence that arbitrary semantic reasoning has been solved.

Observe supports in ten minibatches of 100. Query each minibatch immediately after its writes, before subsequent minibatches are visible. Track the first 100 memories after later writes. Then replace each continuation with a different three-word continuation from the same held-out split, encode the **actually edited support**, query the edited answer, and finally restore the original support. Selection and edits do not depend on model success. The online writer receives only observed token IDs, not QA labels.

Controls include the frozen base, offline-trained/frozen LoRA, contextual Qwen teacher, retrieved-text context using the block reader's candidates, pre-write training-only memory, shuffled V blocks, stale A records when B is correct, and correct-block oracles for both A and B. The retrieved-text baseline observes the same stream minibatches. Main generation uses greedy decoding with a 32-token maximum; report strict normalized EM and token F1, with first-three-word EM only as a diagnostic. Report initial-query candidate recall@32 and selected-block recall@4.

Paired comparisons use matched observation/style IDs. Bootstrap by source article, keeping paraphrases and updates together. Report new-write gains over pre-write/empty memory, gains over shuffled/stale memory, joint A/B/restoration success, and old-memory retention. These confidence intervals describe this held-out sample, not training-seed variation.

The earlier proposed quality criteria remain diagnostic: at least ten EM percentage points over empty and shuffled memory with a positive article-bootstrap lower bound; retrieval recovers at least half of a positive oracle gain; shuffled validation PPL increases by no more than 2%. **No criterion stops the requested full pass.** A PPL improvement without counterfactual write/update gains establishes corpus adaptation, not continual learning.

## Running and artifacts

Install the tested corpus dependencies in [requirements-corpus.txt](../requirements-corpus.txt). Prepare data with [corpus_data.py](../src/vera_mem/corpus_data.py), episodes with [corpus_episodes.py](../src/vera_mem/corpus_episodes.py), and the frozen cache with [corpus_features.py](../src/vera_mem/corpus_features.py). All data, weights, generated examples, raw logs, and checkpoints belong outside Git.

On an authorized compute host, with local paths and an explicitly available GPU:

```bash
PYTHONPATH=src python scripts/run_corpus_round.py \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --data /path/to/prepared_wikitext \
  --episodes /path/to/prepared_episodes \
  --features /path/to/completed_train_features \
  --output /path/to/round_output \
  --gpu GPU-REPLACE-WITH-AN-AVAILABLE-UUID
```

The controller waits for the selected GPU to be free, then builds/qualifies the index, runs a 16-update engineering pilot, trains to the 10M diagnostic, evaluates it, resumes the same pass, runs full confirmation, and performs the [paired analysis](../src/vera_mem/corpus_report.py). Repeated `--candidate-gpu GPU-UUID` arguments permit choosing the first free device from an explicitly authorized list. It chooses one GPU once and launches only one experiment. Availability requires low utilization, at most 1 GiB allocated, and no compute process. It does not terminate another process or use a login/jump host. `--through pilot` stops after engineering qualification. The live controller state, stage receipts, exact commands, source hashes, exit codes, checkpoints, and results are saved under the selected output directory. Remote experimental artifacts are mirrored to local storage and checksum-verified; base-model downloads and environment caches are not duplicated as experiment backups.
