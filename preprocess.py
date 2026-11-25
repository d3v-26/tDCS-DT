"""One-time conversion of the raw NIfTI dataset into a fast training cache.

Why: the original pipeline called nib.load(...).get_fdata() on every volume of
every subject when the Dataset was constructed (float64, ~170 MB per volume,
~180 GB of RAM for the training split) and then ran MONAI transforms on those
float64 arrays in the main process. That takes hours before the first
iteration and keeps the GPU idle during training.

This script reads each subject once and writes plain .npy files:
    <out_dir>/<Split>/<id>_T1.npy        T1, min-max scaled to [0, 1] (same as ScaleIntensityd), float16
    <out_dir>/<Split>/<id>_J_F3F4.npy    E-field magnitude for F3-F4,  float32 (or --label_dtype)
    <out_dir>/<Split>/<id>_J_C3Fp2.npy   E-field magnitude for C3-Fp2, float32 (or --label_dtype)
    <out_dir>/<Split>/<id>_affine.npy    original T1 affine (used to save predictions)
    <out_dir>/<Split>/ids.npy            subjects that converted successfully

The .npy files are opened with mmap at train time, so a worker only reads the
voxels of the crop it needs.

Usage:
    python preprocess.py -root_dir tDCS_DT_data -out_dir cache -workers 16
"""
import argparse
import os
import time
from multiprocessing import Pool

import nibabel as nib
import numpy as np

SPLITS = {'Train': 'train.npy', 'Val': 'val.npy', 'Test': 'test.npy'}
LABELS = {'J_F3F4': 'F3F4', 'J_C3Fp2': 'C3Fp2'}


def _save_atomic(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def convert(job):
    split, sid, root_dir, out_dir, label_dtype, overwrite = job
    src = os.path.join(root_dir, split)
    dst = os.path.join(out_dir, split)
    outputs = [os.path.join(dst, f'{sid}_T1.npy'), os.path.join(dst, f'{sid}_affine.npy')]
    outputs += [os.path.join(dst, f'{sid}_J_{m}.npy') for m in LABELS.values()]
    if not overwrite and all(os.path.exists(p) for p in outputs):
        return split, sid, True, 'cached'
    try:
        t1_img = nib.load(os.path.join(src, 'T1', f'{sid}.nii'))
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        labels = {}
        for folder, name in LABELS.items():
            j = np.asarray(nib.load(os.path.join(src, folder, f'{sid}.nii')).dataobj, dtype=np.float32)
            diff = np.subtract(j.shape, t1.shape)
            if j.shape != t1.shape and np.all(diff == diff[0]) and diff[0] > 0 and diff[0] % 2 == 0:
                # Some ROAST outputs (18 J_C3Fp2 files) keep ROAST's padding of 10 voxels per side
                # (280x331x280 vs 260x311x260). A symmetric centre crop restores voxel alignment
                # (brain-mask IoU vs. J_F3F4 0.998 at offset 10). The original pipeline cropped
                # these with the T1's crop window, i.e. silently misaligned them.
                p = diff[0] // 2
                j = j[p:-p, p:-p, p:-p]
            if j.shape != t1.shape:
                return split, sid, False, f'shape mismatch T1 {t1.shape} vs {folder} {j.shape}'
            labels[name] = j
    except Exception as e:  # corrupted / missing file -> skip subject (original code also skipped these)
        return split, sid, False, f'{type(e).__name__}: {e}'

    lo, hi = float(t1.min()), float(t1.max())
    t1 = (t1 - lo) / (hi - lo) if hi > lo else np.zeros_like(t1)
    _save_atomic(os.path.join(dst, f'{sid}_T1.npy'), t1.astype(np.float16))
    _save_atomic(os.path.join(dst, f'{sid}_affine.npy'), t1_img.affine.astype(np.float64))
    for name, j in labels.items():
        _save_atomic(os.path.join(dst, f'{sid}_J_{name}.npy'), j.astype(label_dtype))
    return split, sid, True, f'shape {t1.shape}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-root_dir', default='tDCS_DT_data', help='raw data with Train/Val/Test sub-folders')
    p.add_argument('-out_dir', default='cache', help='where to write the .npy cache')
    p.add_argument('-workers', type=int, default=8)
    p.add_argument('-label_dtype', default='float32', choices=['float32', 'float16'],
                   help='float16 halves disk/IO; relative error ~5e-4')
    p.add_argument('-splits', nargs='+', default=list(SPLITS))
    p.add_argument('-limit', type=int, default=None, help='only convert the first N subjects per split (debug)')
    p.add_argument('-overwrite', action='store_true')
    args = p.parse_args()

    jobs = []
    for split in args.splits:
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        ids = [str(s) for s in np.load(os.path.join(args.root_dir, split, SPLITS[split]))]
        if args.limit:
            ids = ids[:args.limit]
        jobs += [(split, sid, args.root_dir, args.out_dir, args.label_dtype, args.overwrite) for sid in ids]

    ok = {s: [] for s in args.splits}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for n, (split, sid, success, msg) in enumerate(pool.imap_unordered(convert, jobs), 1):
            if success:
                ok[split].append(sid)
            print(f'[{n}/{len(jobs)} {time.time() - t0:6.0f}s] {split}/{sid}: {"ok" if success else "SKIPPED"} {msg}',
                  flush=True)

    for split in args.splits:
        ids = sorted(ok[split])
        np.save(os.path.join(args.out_dir, split, 'ids.npy'), np.array(ids))
        print(f'{split}: {len(ids)} subjects cached')


if __name__ == '__main__':
    main()
