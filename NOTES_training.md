# Training pipeline: what was changed and why

Log of the 2026-11-25 rework of the tDCS-DT training pipeline (start: commit `b689107`).
Goal: find out why training was so slow and bring it in line with standard practice.

---

## 0. Setup

| What | Detail |
|---|---|
| Data | `tDCS_DT_data -> /blue/ruogu.fang/share/tDCS_DT/tDCS_DT` (symlink, git-ignored). Has `Train/Val/Test`, each with `T1/`, `J_F3F4/`, `J_C3Fp2/` and an id list (`train.npy`, ...). |
| Subjects | Train 354, Val 44, Test 44 (all files present). Two montages each, so 708 / 88 / 88 samples. |
| Volume shapes | 256³ (in-house, T1 `uint8`) and 260×311×260 (HCP, T1 `float32`). Labels are `float32`. |
| Python env | **`envs/tdcs`** (project-local `uv` venv, Python 3.12, git-ignored): torch 2.14.1+cu130, MONAI 1.6.1, nibabel 5.5.0, einops 0.8.2. Exact versions are in `requirements.lock.txt`, and setup commands are at the top of `requirements.txt`. The early benchmarks (3a/3b) used the `IXI_project/envs/crown` env, which has the same torch/MONAI versions. |
| Cluster rules | No compute on login nodes. Check `slurmInfo -u ruogu.fang` before submitting. L4 GPUs (`hpg-turin`) are OK to use; B200s were 100% busy. |

---

## 1. Why the original training was slow (diagnosis)

Measured on the real data:

1. **Everything was loaded eagerly as float64 when the dataset was built.**
   `CustomDataset.load_subjects` called `nib.load(...).get_fdata()` for every T1 and both J maps
   before training started. `get_fdata()` returns float64, about 168 MB per volume.
   - Measured: **76 s for 3 subjects** (cold GPFS read plus float64 conversion), so roughly
     **2.5 h before the first iteration** for the 354 training subjects.
   - RAM: 354 × 3 × 168 MB ≈ **180 GB** for the training split alone.
2. **CPU augmentation in the main process (`num_workers=0`) on float64 256³ arrays.**
   ScaleIntensity, crop, flips and rot90 ran in the training process.
   - Measured: **8.7 s per sample** for transforms alone, so ~1.7 h/epoch (708 samples) with the GPU idle.
3. **Full 256³ input to SwinUNETR, fp32, no AMP.** The stage-1 window attention
   (~6k windows × 343² × 3 heads) needs a single ~9.7 GB allocation. On an L4 (22 GB)
   **every 256³ config OOMs**, even bf16 + gradient checkpointing at batch 1 (benchmark job 52775040).
   On 80 GB GPUs it fits but is very slow. The montage EfficientNet-3D also runs on the full 256³ output.
4. **`torch.cuda.empty_cache()` every iteration** (and `del` of every tensor in `train_old.py`). This forces
   the caching allocator to give memory back and re-request it from the driver every step, which costs throughput.
5. **Per-iteration `print(..., flush=True)` of loss tensors**: forces a GPU→CPU sync every step.
6. **`reg_ece` in numpy.** `np.abs()` / `np.where()` on a CUDA tensor that requires grad raises an error, so the
   script cannot actually run on GPU as written. Even on CPU the result is a constant with **no gradient**,
   so the "ECE regulariser" never affected training. It also copied the full volume to host every step.

### Correctness bugs found along the way (train.py / test.py / utils.py)
- `train()` assigns `best_mae` inside the function: `UnboundLocalError` at the end of epoch 1.
- `validation()` never returns `avg_mae` (returns `None`), and `num_epochs` / `save_path` are undefined in `train()`.
- Subject list paths pointed to `Train_ACT/train.npy` and `Val_ACT/val.npy`, which don't exist.
- Validation used `RandSpatialCropd`, so the validation metric (and model selection) was random.
- `test.py` uses undefined names (`RandSpatialCrop`, `SSIMMetric`, `mscr_loss`, `create_test_dataloader` not imported,
  `x,y,z`), indexes `model(...)[2]` although the model returns 2 outputs, and saves NIfTIs with `np.eye(4)`.
- `requirements.txt`: torch 1.9 / torchvision 0.10, but the code imports `torchvision.models.swin_transformer`
  (needs torchvision ≥ 0.13). MONAI ≥ 1.5 removed `SwinUNETR(img_size=...)`.
- **Data issue: 18 subjects have a padded `J_C3Fp2` map** (280×331×280 vs T1 260×311×260):
  12 Train, 1 Val, 5 Test. The affine origin shifts by exactly 10 voxels × 0.7 mm, which is ROAST's padding.
  In the original pipeline the crop window was computed from the T1 and applied to the larger label, so those
  labels were **silently misaligned by 10 voxels**; at test time the shapes don't match.
  Verified: a 10-voxel symmetric centre crop gives brain-mask IoU **0.998** against that subject's `J_F3F4`
  (0.87 / 0.93 / 0.93 at offsets 0 / 5 / 15).
- **Augmentation semantics:** a left/right flip of a C3-Fp2 field produces a C4-Fp1 field while the label
  vector still says C3-Fp2. 90° rotations produce head orientations that never occur in the data.

---

## 2. What was changed

### `preprocess.py` (new): one-time NIfTI → `.npy` cache
- Reads each subject once (parallel `multiprocessing.Pool`) and writes `cache/<Split>/<id>_T1.npy`
  (min-max scaled to [0,1], identical to `ScaleIntensityd`, **float16**), `<id>_J_F3F4.npy` / `<id>_J_C3Fp2.npy`
  (float32 by default, `-label_dtype float16` optional), `<id>_affine.npy` (T1 affine, used to save predictions),
  and `ids.npy` (subjects that converted).
- Fixes the 18 padded labels with a symmetric centre crop. Skips (and logs) any other mismatch or corrupt file.
- Writes atomically (tmp + rename) and skips finished subjects, so a killed job can simply be re-run.
- **Why:** removes the hours-long startup and the 180 GB RAM need. Each sample becomes 2 small files that are
  memory-mapped, so a worker reads only the voxels of its crop.

### `utils.py`
- `TDCSDataset` replaces `CustomDataset` (no torchio needed). It is lazy and mmap-based: one sample per
  (subject, montage), same vector convention (`[[1],[0]]` = F3F4, `[[0],[1]]` = C3Fp2). Training uses a random crop;
  val/test use the full volume (deterministic).
- DataLoader: `num_workers` (default 8), `pin_memory`, `persistent_workers`, `prefetch_factor=4`, `drop_last` for train.
  T1 is sent as float16 and cast on GPU (half the host→device bytes).
- Spatial augmentation is **off by default** (`-spatial_aug` restores the original flips/rot90; see the L/R note above).
- `MultiModel`: works across MONAI versions (passes `img_size` only if supported). New options:
  `feature_size`, `use_checkpoint`, `use_flash_attention` (MONAI ≥ 1.5 fused attention; avoids the 9.7 GB attention matrix).
  New `predict()` = backbone + head only, used for sliding-window inference so the EfficientNet classifier no longer
  runs on every window. The montage vector is broadcast with `expand` (no copy) and over the batch, so
  `sw_batch_size > 1` works. State-dict keys are unchanged (`backbone.*`, `head.*`, `montage.*`).
- `reg_ece`: torch implementation (bucketize + scatter_add), runs on GPU and is **differentiable** through the per-bin
  errors (bin membership treated as constant). Empty bins are ignored. The original counted them as 0 and dropped the
  last bin through `np.diff(...)[:-1]`.
- `vector_class`: simplified to `argmax`.

### `train.py` (rewritten; original flag names kept)
- bf16 autocast (default; `-amp fp16` uses GradScaler, `-amp none` is fp32), TF32, `cudnn.benchmark`.
- No `empty_cache()`. Losses are accumulated on the GPU and synced only every `-log_every` steps. The log line shows
  s/it and time waiting on data, so input bottlenecks are visible.
- AdamW (lr 1e-4, wd 1e-5) with linear warmup + per-step cosine decay, gradient clipping 1.0. The original was Adam
  with a constant lr of 1e-3, which is high for a Swin transformer.
- Same loss: `L1 + 0.5·CE + 0.5·ECE` (weights are now flags `-w_class`, `-w_ece`).
- Validation every `-val_every` epochs (default 2), full-volume sliding window (gaussian blending, overlap 0.25).
  Saves the best `state_dict` to `-save_path` and a full resumable checkpoint to `*_last.pth` every epoch (`-resume`).
- Optional `-use_checkpoint`, `-flash_attention`, `-compile`. `-limit_batches` is for smoke tests.
- `-root_dir` now points to the **cache** directory (default `cache`).

### `test.py` (rewritten)
- Uses the cache. Full-volume sliding window. Metrics: MAE, MSE, SSIM, ECE (replaces the undefined `mscr_loss`),
  montage accuracy (classifier on the centre `roi` crop). Labels are clipped at 3.0 as in the original.
  Results are reported for All / Old (256³) / Young (HCP) as before. Predictions are saved with the real T1 affine.

### SLURM / misc
- `slurm/preprocess.sbatch` (hpg-default, 16 CPU, 96 GB) and `slurm/train.sbatch` (hpg-turin, `gpus=l4:1`, 12 CPU).
- `.gitignore`: data link, `cache/`, `Models/`, `Results/`, `logs/`, `*.pth`.
- `logs/bench_gpu.py` + `logs/bench.sbatch`: per-config GPU step-time and memory benchmark.

---

## 3. Benchmarks

### 3a. Data pipeline (real data)
| | Original | New |
|---|---|---|
| Dataset construction | ~25 s/subject (eager `get_fdata`, float64), so ~2.5 h + ~180 GB RAM for Train | instant (lazy mmap; one-time `preprocess.py` job instead) |
| Per-sample CPU work | 8.7 s (float64 MONAI transforms, main process) | crop from mmap in 8 workers. Measured **data wait ≈ 0.01 s/it** in training (smoke job 52778638, L4) |

### 3b. One training step (forward + backward + AdamW; random tensors, full MultiModel incl. EfficientNet, `logs/bench_gpu.py`)

**RTX PRO 6000 Blackwell, 96 GB** (jobs 52778637, 52779442):

| Config | s/it | s/sample | peak GiB |
|---|---|---|---|
| 256³ fp32 bs2 (**original config**), with or without empty_cache | OOM | | |
| 256³ fp32 bs1 | OOM | | |
| 256³ bf16 bs2 (± flash, ± checkpointing) | OOM | | |
| **256³ bf16 bs1** | **1.18** | 1.18 | 74.0 |
| 256³ bf16 bs1 + grad checkpointing | 1.35 | 1.35 | 41.2 |
| 192³ bf16 bs1 | 0.48 | 0.48 | 30.2 |
| 192³ bf16 bs2 | 0.82 | 0.41 | 60.1 |
| 128³ fp32 bs2 | 0.32 | 0.16 | 38.5 |
| **128³ bf16 bs2** | **0.27** | 0.13 | 20.5 |
| 128³ bf16 bs2 + grad checkpointing | 0.31 | 0.16 | 11.8 |
| 128³ bf16 bs8 | 0.94 | 0.12 | 81.4 |

**NVIDIA L4, 22 GB** (jobs 52775040, 52775623, 52776545): every 256³, 192³ and 160³ config OOMs, and 128³ fits only at bs1
(backbone alone: 8.2 GiB at bs1, 16.4 GiB at bs2). The 256³ failure is the 9 GiB softmax of the stage-1 Swin window attention.
Real training at 128³ bs1 on an L4: **0.66 s/it** (smoke job 52778638).

**Takeaways**
- The original setup (256³, fp32, batch 2) **does not fit on any GPU available here**, not even 96 GB. bf16 autocast is
  what makes full-volume training possible at all.
- `use_flash_attention` in MONAI 1.6.1 SwinUNETR made **no measurable difference** (identical 74.0 GiB and 1.18 s/it).
  The flag stays in the code but is not recommended.
- Estimated epoch time (708 samples, without validation):
  - 256³ bf16 bs1 on one RTX PRO 6000: ~14 min/epoch, so ~28 h for 120 epochs. **About 7 h on 4 GPUs with DDP.**
  - 128³ bf16 bs2 on one RTX PRO 6000: ~1.6 min/epoch. On an L4 at bs1: ~7.8 min/epoch.

### 3c. Multi-GPU
`train.py` supports DDP via `torchrun`. `-batch_size` is per GPU. The training set is sharded with `DistributedSampler`,
validation is split across ranks and all-reduced, and only rank 0 logs and saves.
Example: `python -m torch.distributed.run --standalone --nproc_per_node=4 train.py ...`

**Verified (job 52780102, 2× L4, 128³ bs1/GPU):** 2-epoch DDP run, then `-resume` from `*_last.pth` to epoch 3,
then `test.py` on the saved weights. All three ran without errors. Val MAE went down each epoch and best-checkpoint saving
worked. Steady state was ~0.7 s/it per rank, with data wait ≤ 0.05 s/it.

### 3d. Full cache build
`preprocess.py` on all splits (job 52779487, hpg-milan): **22 s wall** for 442 subjects, 0 skipped or failed.
Train 354 / Val 44 / Test 44 cached. `logs/verify_cache.py` (job 52780582) re-read 4 subjects from NIfTI:
T1 max error 2.4e-4 (float16 rounding), both J maps bit-identical (including a padded C3Fp2 that was centre-cropped),
label file counts 708 / 88 / 88, no `.tmp` leftovers.

---

## 4. Open decisions / next steps
- ~~Crop size~~: **decided 256³ (full volume) on RTX PRO 6000**, see section 5.
- **Crop size on L4.** 256³ does not fit on 22 GB without flash attention. Smaller crops (128³ to 192³) are the
  standard SwinUNETR setting, but the E-field depends on whole-head geometry, so patches lose global context.
  Needs an accuracy comparison, not just speed.

---

## 5. Production run setup (2026-11-25)

**Env.** Created `envs/tdcs` with `uv` on a compute node. `einops` had to be added by hand: MONAI's SwinUNETR imports it
through `optional_import`, so `pip install monai` does not install it, and the first smoke job (52782403) failed with
`No module named 'einops'`. It is now pinned in `requirements.txt`. Training does not need torchio or torchvision.
Only `train_old.py` imports them.

**GPUs: how many RTX PRO 6000s?** One is enough for 256³ (bf16, bs1: 74 GiB of 96). Batch size 2 per GPU OOMs even on
96 GB, so global batch = number of GPUs. **2 GPUs were chosen:** global batch 2 matches the original setting, and it is
about 1.7× faster than one GPU. 4 GPUs would double the global batch to 4 and need an lr re-check, for diminishing returns.

**`slurm/train.sbatch`** now targets `hpg-rtx6000`, `--gpus=rtx_pro_6000:2`, 16 CPU, 128 GB, 3 days. It launches through
`torch.distributed.run` with nproc = number of allocated GPUs (`--gpus=rtx_pro_6000:1` also works), and runs
256³, bs1/GPU, 120 epochs, with `envs/tdcs` as the default Python. `slurm/preprocess.sbatch` also uses `envs/tdcs` now.

**Smoke test at 256³ on real data** (job 52782463, 2× RTX PRO 6000, 30 steps + val limited to 30 samples/rank):
1.50 s/it at iteration 30 and still falling after warmup (single-GPU benchmark: 1.18 s/it). Data wait 0.03 s/it.
Validation took 66 s for 30 samples per rank, so about 1.6 min for the full 88 samples.
Estimate: 354 steps × ~1.4 s ≈ 8.5 min/epoch, plus validation every 2 epochs, gives **~18 h for 120 epochs**.

**Full run submitted: job 52782720** → `Models/best_model.pth` (best val MAE) and `Models/best_model_last.pth`
(resumable every epoch). Log: `logs/tdcs_train_52782720.out`.
