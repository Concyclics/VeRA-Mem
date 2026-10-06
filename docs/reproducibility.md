# Reproducibility and artifact provenance

## Choose the intended operation

| Goal | Entry point | What it establishes |
| --- | --- | --- |
| Read the current evidence | [Research path](research_path.md), [block results](block_results.md) | Reported outcomes and their limits |
| Validate this English handoff | `python scripts/check_repository.py` | Language coverage, local document links, and localization provenance |
| Check implementation contracts | `python -m pytest -q` | CPU storage/data/gradient/evaluation contracts; not model capability |
| Try the current block pipeline | The explicit smoke commands below | A new local execution, not a replication of the completed study |
| Recompute historical results | Archived code at the pinned revision plus original run artifacts | Recomputed aggregates or replayed evidence, depending on the auditor |
| Start a new scientific comparison | New protocol, data, plans, output directories, and registration | A separately identified experiment with fresh confirmation evidence |

## Environment and storage

The tested software versions are in [tested_environment.json](tested_environment.json). The package declares broader compatible ranges, so resolving dependencies today is not itself an exact environment reproduction. Use a CUDA-compatible PyTorch build on the selected compute device. Tests and evidence inspection can run on CPU.

The completed experiments used `xtrah100`, account `chenhan`, and `/ssd3/chenhan/VeRA-Mem-Workspace`. The local verified mirror was `/mnt/storage/chenhan/xtraNet/VeRA-Mem-Workspace`. These are provenance locations, not assumptions a new collaborator must reproduce. No credentials are in this repository.

```text
VeRA-Mem-Workspace/
  VeRA-Mem/                         # This Git repository
  models/
    manifest.json                   # Pinned base-model revision
    Qwen3-4B-Instruct-2507/          # Download separately
  data/                             # Prepared data and feature caches
  runs/                             # New runs on the compute host
    xtrah100/                       # Historical run mirror on the original local host
  plans/                            # Registration, job plans, completion receipts
  remote_snapshots/xtrah100/source/ # Mirrored source, outside the code repository
```

Raw features, checkpoints, logs, token traces, and immutable execution snapshots are not included in Git. Request the relevant run artifacts from the project maintainer to reproduce their audits. Original student materials, private attachments, SSH settings, base weights, and environment caches are outside the public repository. The public JSON contains synthetic examples, aggregate/paired measurements, hashes, and some original machine paths; it does not constitute a pretrained model distribution.

## Install and prepare the pinned model

From the repository directory:

```bash
python3 -m venv ../env
. ../env/bin/activate
python -m pip install -e '.[test,data]'
python -m pytest -q

python scripts/prepare_model.py --workspace .. \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554
```

Model preparation resolves and records the revision in `../models/manifest.json`. It checks for at least 13 GiB of free disk space before downloading; features and experiment artifacts require additional space. Model terms and access are governed by the [official model repository](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Verify GPU availability, free host memory, and target filesystem space before choosing a device. A resource snapshot is not a reservation. These commands do not terminate other processes:

```bash
nvidia-smi --query-gpu=uuid,name,memory.used,memory.free,utilization.gpu --format=csv
df -h ../
export CUDA_VISIBLE_DEVICES='GPU-REPLACE-WITH-A-CONFIRMED-AVAILABLE-UUID'
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

On a scheduler-managed system, run training and model inference only inside allocated compute resources. Do not run them on login, account-management, or network-jump hosts.

## Current block pipeline: a smoke example

Choose unused output paths; runners refuse to overwrite existing directories. These examples prepare the fixed tiny dataset and run only two training updates. Feature preparation still encodes 8,192 contextual training observations plus evaluation views; it requires the real model and is not a zero-cost unit test.

```bash
python -m vera_mem.block_run --stage prepare \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --data-seed 221042 --run-dir ../runs/block_demo_features

python -m vera_mem.block_run --stage teacher \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --cache ../runs/block_demo_features/train.pt \
  --eval-part preflight --run-dir ../runs/block_demo_teacher

python -m vera_mem.block_run --stage train \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --cache ../runs/block_demo_features/train.pt \
  --block-mode block_outer --regime rebind --seed 91042 --updates 2 \
  --run-dir ../runs/block_demo_train

python -m vera_mem.block_run --stage eval \
  --model ../models/Qwen3-4B-Instruct-2507 \
  --cache ../runs/block_demo_features/known.pt \
  --checkpoint ../runs/block_demo_train/last.pt \
  --eval-part smoke --run-dir ../runs/block_demo_eval
```

This reuses already observed research data for pipeline verification. It does not provide new confirmation evidence, and a two-update checkpoint should not be judged against the formal results. For the completed 2,048-update study, consult [plans.json](results/block/plans.json) and [block_protocol.md](block_protocol.md): three modes, two binding regimes, three seeds, fixed endpoints, controls, teacher checks, and separate known/new-entity evaluation. Historical plan generators and orchestrators contain original paths, dates, and GPU assignments; create new plans rather than launching them blindly.

## English localization and sealed evidence

The authoritative pre-localization repository revision is:

```text
768460245498b94a375b2bd06060fa353a468f17
```

The current English documents are translations and reorganized explanations of that record. The [English localization manifest](localization_manifest.json) binds each changed historical file to its original and current SHA256 and enumerates localized JSON fields. Numerical JSON values, booleans, array structure, IDs, predictions, and historical hash strings are preserved. The `teaching_point_zh` key in the historical example schema is retained for compatibility; its human explanation is now English. The translated reconstruction protocol diff is an explanatory rendering, not a byte-exact patch against the original document.

Historical registrations, audit receipts, source hashes, and selection decisions are not resealed to match translated text. Their hashes continue to identify the original artifacts. This also applies to historical summary script hashes: translating a report renderer does not mean old results were generated by the translated source. Raw run snapshots and original Git history remain untouched.

**Strict historical validators may reject translated protocols in the current checkout. This is expected, not a reason to weaken their checks.** Audit old experiments from the pinned revision or their immutable run snapshots. New experiments need new registration receipts binding their current English protocols and runtime sources.

For example, extract a disposable copy of the original repository without changing your working checkout:

```bash
export VERA_EVIDENCE_COMMIT=768460245498b94a375b2bd06060fa353a468f17
export VERA_AUDIT_DIR="$(mktemp -d /tmp/vera-evidence.XXXXXX)"
git archive "$VERA_EVIDENCE_COMMIT" | tar -x -C "$VERA_AUDIT_DIR"
export VERA_RUNS_ROOT='/ABSOLUTE/PATH/TO/HISTORICAL/runs/xtrah100'
export VERA_AUDIT_OUTPUT="$(mktemp -d /tmp/vera-audit-output.XXXXXX)"

PYTHONPATH="$VERA_AUDIT_DIR/src" python "$VERA_AUDIT_DIR/scripts/summarize_block.py" \
  --runs-root "$VERA_RUNS_ROOT" \
  --output "$VERA_AUDIT_OUTPUT/summary.json" --require-final

PYTHONPATH="$VERA_AUDIT_DIR/src" python "$VERA_AUDIT_DIR/scripts/audit_block.py" \
  --runs-root "$VERA_RUNS_ROOT" \
  --registration "$VERA_AUDIT_DIR/docs/results/block/registration.json" \
  --output "$VERA_AUDIT_OUTPUT/independent_audit.json" --require-final
```

The audit output file must not already exist. The auditor needs CPU PyTorch and complete original run artifacts, but does not load or regenerate the 4B model. It reconstructs recorded banks, writes/restores, and all saved query routes. It does not independently establish that each query was produced by a new model execution. Earlier rounds have different telemetry and audit scope; do not transfer the block study's full-query replay guarantee to them.

## Results, costs, and backups

The final block study has 96 formal tasks and six separate smoke tasks. Its formal record includes 21,600 student generations, 448 teacher generations, 106,565 student-generated tokens/queries, and 8,064 logical group writes/restores. Initialization fill and control-bank materialization are excluded from that write count. Concurrent process-duration sums are not exclusive GPU wall time.

The historical test receipt records 1,249 full-repository pytest passes, followed by 85 distinct registry/summary tests, plus 14 auditor and 10 mechanism-diagnostic self-tests. This is not one combined run of 1,334 tests. Current handoff validation is recorded separately in [handoff_validation.json](handoff_validation.json).

The authorized original-host backup command is:

```bash
python scripts/sync_backup.py --host xtrah100 \
  --remote-root /ssd3/chenhan/VeRA-Mem-Workspace --local-root .. --verify
```

It mirrors run artifacts, data, source, and the model manifest, excluding base weights and environment caches. A checksum check while jobs are writing is not a final snapshot; completion requires finished jobs and empty checksum differences. A collaborator should substitute their authorized host/workspace or transfer the relevant artifact set through an agreed channel.
