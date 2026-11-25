import numpy as np, nibabel as nib, glob, os
r, c = 'tDCS_DT_data', 'cache'
for split, sid in [('Train', '101107'), ('Train', '100206'), ('Val', '106521'), ('Test', '178950')]:
    t1 = np.asarray(nib.load(f'{r}/{split}/T1/{sid}.nii').dataobj, dtype=np.float32)
    t1 = (t1 - t1.min()) / (t1.max() - t1.min())
    ct1 = np.load(f'{c}/{split}/{sid}_T1.npy')
    jf = np.asarray(nib.load(f'{r}/{split}/J_F3F4/{sid}.nii').dataobj, dtype=np.float32)
    jc = np.asarray(nib.load(f'{r}/{split}/J_C3Fp2/{sid}.nii').dataobj, dtype=np.float32)
    if jc.shape != t1.shape: jc = jc[10:-10, 10:-10, 10:-10]
    print(split, sid, ct1.shape, ct1.dtype, 'T1 maxerr %.1e' % np.abs(ct1.astype(np.float32) - t1).max(),
          'F3F4 equal', np.array_equal(np.load(f'{c}/{split}/{sid}_J_F3F4.npy'), jf),
          'C3Fp2 equal', np.array_equal(np.load(f'{c}/{split}/{sid}_J_C3Fp2.npy'), jc), flush=True)
for s in ['Train', 'Val', 'Test']:
    n = len(np.load(f'{c}/{s}/ids.npy')); f = len(glob.glob(f'{c}/{s}/*_J_*.npy'))
    print(s, 'ids', n, 'label files', f, 'tmp leftovers', len(glob.glob(f'{c}/{s}/*.tmp.npy')))
