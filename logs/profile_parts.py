import sys, torch, traceback
sys.path.insert(0, '.')
from utils import MultiModel
dev = 'cuda'
roi = int(sys.argv[1]); bs = int(sys.argv[2])
m = MultiModel(1, 2, 1, roi, roi, roi).to(dev)
x = torch.rand(bs, 1, roi, roi, roi, device=dev)
v = torch.zeros(bs, 2, 1, device=dev); v[:, 0] = 1
for name, fn in [('backbone+head', lambda: m.predict(x, v)), ('efficientnet', lambda: m.montage(x))]:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = fn()
        out.float().mean().backward()
        print(f'PROF roi {roi} bs {bs} {name:14s} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
    except torch.OutOfMemoryError as e:
        print(f'PROF roi {roi} bs {bs} {name:14s} OOM: {str(e)[:110]}', flush=True)
        tb = traceback.extract_tb(e.__traceback__)
        print('PROF   at', ' <- '.join(f'{f.name}:{f.lineno}' for f in tb[-4:][::-1]), flush=True)
    m.zero_grad(set_to_none=True); out = None
