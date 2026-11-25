"""Throughput/memory benchmark of one training step at 256^3 on the current GPU."""
import sys, time, torch, torch.nn as nn
sys.path.insert(0, '.')
from utils import MultiModel, reg_ece

dev = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def bench(name, roi, bs, amp, ckpt, flash=False, empty_cache=False, steps=6):
    torch.cuda.reset_peak_memory_stats()
    try:
        m = MultiModel(1, 2, 1, *roi, use_checkpoint=ckpt, use_flash_attention=flash).to(dev)
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
        x = torch.rand(bs, 1, *roi, device=dev); y = torch.rand(bs, 1, *roi, device=dev)
        v = torch.zeros(bs, 2, 1, device=dev); v[:, 0] = 1; c = torch.zeros(bs, dtype=torch.long, device=dev)
        ts = []
        for i in range(steps):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                out, cls = m(x, v)
            out = out.float()
            loss = nn.functional.l1_loss(out, y) + .5 * nn.functional.cross_entropy(cls.float(), c) + .5 * reg_ece(out, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if empty_cache: torch.cuda.empty_cache()
            torch.cuda.synchronize(); ts.append(time.time() - t)
        s = sum(ts[2:]) / len(ts[2:])
        print(f'RESULT {name:52s} {s:6.2f} s/it  {s / bs:6.2f} s/sample  peak {torch.cuda.max_memory_allocated() / 2**30:5.1f} GiB', flush=True)
    except torch.OutOfMemoryError:
        print(f'RESULT {name:52s} OOM', flush=True)

# usage: bench_gpu.py NAME ROI BS AMP(0/1) CKPT(0/1) FLASH(0/1) EMPTYCACHE(0/1)
a = sys.argv[1:]
bench(a[0], (int(a[1]),) * 3, int(a[2]), a[3] == '1', a[4] == '1', a[5] == '1', a[6] == '1')
