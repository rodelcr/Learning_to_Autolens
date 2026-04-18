---
name: fasrc-cannon-gpu
description: Harvard FASRC Cannon 2.0 cluster guide for GPU computation with SLURM. Covers job submission, GPU allocation, OOM management, parallel sharding, fairshare strategy, and common pitfalls. Consult when writing .sbatch scripts, debugging GPU jobs, or planning multi-GPU workflows.
---

# FASRC Cannon 2.0 — GPU Computation Guide

Hard-won lessons from running LLM fine-tuning, inference, and evaluation jobs on Harvard's Cannon cluster. Applicable to any GPU-heavy workload (ML training, scientific computing, etc.).

## Cluster Environment

### Basics
- **Login:** `ssh USERNAME@login.rc.fas.harvard.edu`
- **Home:** `/n/home02/USERNAME/` — limited quota, keep code only
- **Large storage:** Use lab storage under `/n/holylfs*/LABS/` or `/n/holystore*/LABS/` for checkpoints, datasets, model weights
- **Module system:** `module load cuda/12.4` (always load before GPU work)
- **Conda:** FASRC provides a shared Miniforge installation:
  ```bash
  source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
  conda activate YOUR_ENV
  ```
- **Web-accessible output:** Some labs have web-viewable directories under `/n/holylfs*/LABS/*/Everyone/*/www/`

### SLURM Accounts

**CRITICAL:** Always specify `--account=YOUR_GROUP` on every `sbatch` and `salloc` submission. Omitting it may charge fairshare to the wrong group, tanking your priority where you don't intend.

Check your accounts: `sacctmgr show associations where user=$USER format=Account`

### GPU Partitions

| Partition | Scheduling | Max Time | Notes |
|-----------|-----------|----------|-------|
| `gpu` | Slow (guaranteed) | varies | Long queue wait, but jobs won't be preempted |
| `gpu_requeue` | Medium | varies | **Preemptible** — your job can be killed anytime. Must checkpoint frequently |
| `gpu_test` | Fast | 1 hour | Quick smoke tests only |
| Lab-specific (e.g. `siag_gpu`) | Varies | varies | Check with your PI — some labs have dedicated partitions with faster scheduling |

### GPU Selection
- **Specific GPU type:** `--gres=gpu:nvidia_a100-sxm4-80gb:1` (most precise — specifies exact model)
- **Any GPU:** `--gres=gpu:1`
- **Constraint (less flexible):** `--constraint=a100` — limits to A100 nodes but doesn't specify VRAM size. Can cause longer queue waits.
- **Always verify what you got:** `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`

List available GPU types: `sinfo -p gpu -o "%G %N" | sort -u`

## SBATCH Script Template

```bash
#!/bin/bash
#SBATCH -J jobname              # Short name (appears in squeue)
#SBATCH -p gpu                  # Partition (gpu, gpu_requeue, gpu_test, or lab-specific)
#SBATCH --account=YOUR_GROUP    # ALWAYS specify — see note above
#SBATCH -N 1                    # Nodes (almost always 1 for single-node GPU work)
#SBATCH -n 4                    # CPU cores (4 for inference, 8-16 for training)
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --mem=200G              # System RAM (not VRAM)
#SBATCH -t 0-02:00              # Wall time: D-HH:MM (be generous but not wasteful)
#SBATCH -o logs/%x.%j.out       # Stdout (%x=job name, %j=job ID)
#SBATCH -e logs/%x.%j.err       # Stderr
#SBATCH --mail-type=END,FAIL    # Email on completion or failure
#SBATCH --mail-user=YOUR_EMAIL

# === Environment setup ===
module load cuda/12.4
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate YOUR_ENV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1       # Flush stdout immediately (essential for log monitoring)
cd $HOME/YOUR_PROJECT
mkdir -p logs                   # SLURM fails silently if output dir doesn't exist

# === Diagnostics (always include — invaluable for debugging) ===
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA: $(nvcc --version | tail -1)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Date: $(date)"
echo ""

# === Your work here ===
python your_script.py --args

echo "=== Done at $(date) ==="
```

## Resource Sizing Guide

Empirically tested sizes for LLM workloads on A100 80GB:

| Task | GPUs | Mem | CPUs | Time | Notes |
|------|------|-----|------|------|-------|
| 70B QLoRA training (seq_len=1024) | 1 | 250G | 16 | 14h | ~26.5 sec/step, single A100 80GB |
| 70B QLoRA training (seq_len=2048) | 2 | 250G | 16 | 14h | Requires DeepSpeed ZeRO-3 (see OOM section) |
| 70B 4-bit inference/eval | 2 | 200G | 4 | 2h | Per evaluation shard of ~90 items |
| 8B QLoRA training | 1 | 80G | 8 | 4h | Comfortable on single A100 |
| Embedding model (e5-large etc.) | 1 | 64G | 4 | 2h | Light GPU usage |
| Data preprocessing (no GPU) | 0 | 16-32G | 4 | varies | Submit to `shared` partition instead |

**Rule of thumb:** Request 3-4x the model's parameter size in system RAM (not VRAM). A 70B model in 4-bit needs ~35GB VRAM but benefits from 200-250G system RAM for data loading and preprocessing.

## Parallel Job Sharding

### The Problem
Many workloads (evaluation, data processing) can be trivially parallelized across items, but:
- Shared resources (SQLite databases, file locks) crash with concurrent access
- Sequential processing is too slow for large datasets

### The Solution: Precompute + Shard

**Step 1: Precompute shared resources into a static file (single job)**
```bash
# Avoids concurrent access to databases/APIs during parallel phase
python precompute.py --output data/precomputed_vN.json
```

**Step 2: Launch parallel shards (N independent jobs)**
```bash
# Each shard reads from the precomputed file, processes a slice
python evaluate.py \
    --start 0 --end 100 --shard-id shard_0 \
    --precomputed data/precomputed_vN.json \
    --output results/
```

**Step 3: Merge results**
```bash
python merge_results.py results/eval_shard_*.json --output results/eval_combined.json
```

### SLURM Arrays vs Individual Jobs
Both work. Arrays (`#SBATCH --array=0-7`) are cleaner but schedule slower on FASRC than separate .sbatch files. Use individual files for production runs where scheduling speed matters.

```bash
# Array approach (single submit, slower scheduling):
#SBATCH --array=0-7
SHARD_SIZE=$((TOTAL / 8))
START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
END=$(((SLURM_ARRAY_TASK_ID + 1) * SHARD_SIZE))

# Individual files (multiple submits, faster scheduling):
for i in $(seq 0 7); do sbatch eval_shard_${i}.sbatch; done
```

### Version Everything
- Always version output files: `results_v3.json`, not `results.json`
- Never overwrite previous versions — you will need them for comparison
- If results depend on precomputed data, the versions must match

## OOM Management

### Large Model QLoRA on A100 80GB

| seq_len | GPUs | Status | Notes |
|---------|------|--------|-------|
| 512 | 1 | Safe | Conservative, may truncate useful context |
| 1024 | 1 | Reliable | Sweet spot for single-GPU 70B QLoRA |
| 2048 | 1 | **Fails** | OOMs around step ~170 (delayed OOM from long examples) |
| 2048 | 2 | Works | Requires DeepSpeed ZeRO-3 |

### Key facts
- **Delayed OOM is real:** A job can run fine for hours, then hit one unusually long training example and crash. Always test with `--max-steps 10` before committing to a long run.
- **`device_map="auto"` does NOT work for multi-GPU training.** It shards the model across GPUs but causes cross-device tensor errors during loss computation. Use DeepSpeed ZeRO-3 instead.
- **`device_map="auto"` DOES work for multi-GPU inference.** Safe for evaluation/generation.

### Prevention
```bash
# Always set in your sbatch scripts:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Monitor VRAM during development:
watch -n 5 nvidia-smi
# Or in a script:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 30
```

### Recovery
1. Your last checkpoint (saved every N steps) is safe — resume from it
2. To fix: reduce `seq_len` > `batch_size` > `grad_accum` (in that priority order)
3. Alternative: truncate long examples in preprocessing rather than reducing seq_len globally

## Fairshare Management

### How it works
- FASRC uses fairshare to prioritize jobs — heavy recent usage lowers your priority
- Priority recovers over time (days, not hours)
- `sshare -u $USER` shows current fairshare scores across your accounts

### Strategies
- **Spread jobs across days** when possible — don't submit 20 GPU jobs in one afternoon
- **Use `gpu_requeue`** for non-critical work — lower fairshare cost since preemptible jobs are "cheaper"
- **Use `gpu_test`** (1hr limit) for smoke tests — minimal fairshare impact
- **Rotate accounts** if you belong to multiple groups — balance usage across them
- **Monitor:** `squeue -u $USER` to see running/pending jobs, `sprio -u $USER` for priority scores

## Interactive Debugging with `salloc`

For iterative development, `salloc` is far faster than the submit-wait-read-logs-fix-resubmit cycle:

```bash
# Request an interactive GPU session
salloc -p gpu_test --account=YOUR_GROUP --gres=gpu:1 --mem=80G -t 1:00:00

# Once allocated, you're on the compute node:
module load cuda/12.4
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate YOUR_ENV
python -c "import torch; print(torch.cuda.get_device_name())"

# Now iterate freely — edit code on login node, run on compute node
python train.py --max-steps 5  # quick test
python train.py --max-steps 5 --seq-len 2048  # test longer sequences
```

### Debugging Checklist
When a job fails or behaves unexpectedly:
1. **GPU available?** `nvidia-smi` — check VRAM, utilization, existing processes
2. **CUDA loaded?** `module list | grep cuda` — must be loaded in every script
3. **Right Python?** `which python` — should point to your conda env, not system Python
4. **Env vars set?** `echo $PYTORCH_CUDA_ALLOC_CONF` — must be set in the script, not inherited
5. **Working directory?** `pwd` — scripts assume relative paths from project root
6. **Output dirs exist?** SLURM fails silently if `-o` or `-e` directories don't exist

## HuggingFace Offline Mode

Cannon compute nodes may not have internet access. Once models are cached locally (downloaded from a login node), prevent network calls during jobs:

```bash
export HF_HOME=$HOME/.cache/huggingface  # or your preferred cache location
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**First download:** Run on a login node (which has internet):
```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.3-70B-Instruct')"
```

## Multi-Step Pipeline Pattern

For workflows with sequential dependencies (e.g., preprocess -> train -> evaluate), combine steps in a single script with conditional guards:

```bash
# Step 0: Conditional preprocessing (skip if already done)
if [ ! -f data/processed/train_augmented.jsonl ]; then
    echo "=== Preprocessing ==="
    python scripts/preprocess.py
fi

# Step 1: Train
echo "=== Training ==="
python train.py --data data/processed/train_augmented.jsonl --output checkpoints/

# Step 2: Evaluate (depends on Step 1)
echo "=== Evaluating ==="
python evaluate.py --model checkpoints/final/ --output results/
```

Use a single script for pipelines under ~4 hours. For longer workflows, split into separate jobs with `--dependency=afterok:JOBID`:
```bash
JOB1=$(sbatch --parsable preprocess.sbatch)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 train.sbatch)
sbatch --dependency=afterok:$JOB2 evaluate.sbatch
```

## Auto-Commit Pattern for Unattended Jobs

Track results in git automatically at the end of long-running jobs:

```bash
# At the end of your sbatch script:
cd $HOME/YOUR_PROJECT
if [ -n "$(git status --porcelain results/)" ]; then
    git add results/*.json
    git commit -m "Results from $(hostname) $(date +%Y-%m-%d)" \
        --author="Cluster Job <noreply@fas.harvard.edu>" || true
    git push || echo "Warning: git push failed (network may be unavailable)"
fi
```

- `|| true` on commit: may have nothing new to commit
- Warn-only on push: compute nodes may lack network access
- Consider `--author` to distinguish automated commits from manual ones

## SQLite on Shared Filesystems

Many tools (ChromaDB, MLflow, DVC) use SQLite internally. SQLite does NOT handle concurrent access well on networked filesystems (NFS/Lustre):

- **Single reader/writer:** Works fine
- **Multiple readers:** May work but can produce `database is locked` errors
- **Multiple writers:** Will corrupt the database

### Workarounds
1. **Precompute:** Export SQLite data to JSON/Parquet before parallel jobs (recommended)
2. **Copy to local SSD:** `cp db.sqlite3 /tmp/db.sqlite3` at job start, copy back after
3. **Use a different backend:** PostgreSQL, or flat files for simple cases
4. **Serialize access:** Run database-dependent work in a single preprocessing job

## Common Pitfalls

1. **Forgetting `--account`** — Job runs under the wrong group, wrecking fairshare for a PI you didn't intend to charge
2. **Output directory doesn't exist** — SLURM silently refuses to start the job. Always `mkdir -p logs` before or in your script
3. **Stale environment** — Module loads and conda activations do NOT carry from login node to compute node. Set them explicitly in every sbatch script
4. **Delayed OOM** — Job runs fine for hours, then one long example triggers OOM. Always smoke-test first
5. **Preemption without checkpointing** — `gpu_requeue` jobs can be killed at any time. Save checkpoints frequently (every N steps)
6. **`device_map="auto"` for training** — Works for inference, fails for training with cross-device tensor errors. Use DeepSpeed ZeRO-3 for multi-GPU training
7. **`--constraint=a100` vs `--gres=gpu:nvidia_a100-sxm4-80gb:1`** — The constraint is broader (any A100) but may limit scheduling. The gres is more precise
8. **Buffered stdout** — Python buffers output, so log files appear empty during long runs. Always set `export PYTHONUNBUFFERED=1` or use `python -u`
9. **SQLite on shared filesystems** — Concurrent access crashes. Precompute to JSON instead (see SQLite section)
10. **Forgetting to verify GPU VRAM** — You requested an A100 80GB but got an A100 40GB. Always print `nvidia-smi` output at job start
