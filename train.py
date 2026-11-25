#Load Standard Libraries
import argparse
import math
import os
import time

#Load Torch Libraries
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

#Load Monai Packages
from monai.inferers import SlidingWindowInferer

#Load Custom
from utils import create_dataloaders, MultiModel, reg_ece


def get_args():
    parser = argparse.ArgumentParser()
    # --- original arguments (same names) ---
    parser.add_argument('-x', type=int, default=256, help='crop size, axis 0 (must be divisible by 32)')
    parser.add_argument('-y', type=int, default=256, help='crop size, axis 1')
    parser.add_argument('-z', type=int, default=256, help='crop size, axis 2')
    parser.add_argument('-batch_size', type=int, default=2, help='batch size per GPU')
    parser.add_argument('-root_dir', type=str, default='cache', help='cache directory written by preprocess.py')
    parser.add_argument('-in_channels_img', type=int, default=1)
    parser.add_argument('-in_channels_vector', type=int, default=2, help='number of electrode montages')
    parser.add_argument('-out_channels', type=int, default=1)
    parser.add_argument('-num_epochs', type=int, default=120)
    parser.add_argument('-save_path', type=str, default='Models/best_model.pth', help='best model (state_dict)')
    # --- loss weights (original: L1 + .5*CE + .5*ECE) ---
    parser.add_argument('-w_class', type=float, default=0.5)
    parser.add_argument('-w_ece', type=float, default=0.5)
    # --- optimisation ---
    parser.add_argument('-lr', type=float, default=1e-4, help='original used Adam 1e-3')
    parser.add_argument('-weight_decay', type=float, default=1e-5)
    parser.add_argument('-warmup_epochs', type=float, default=2)
    parser.add_argument('-grad_clip', type=float, default=1.0, help='<=0 disables')
    parser.add_argument('-amp', choices=['bf16', 'fp16', 'none'], default='bf16')
    parser.add_argument('-use_checkpoint', action='store_true', help='gradient checkpointing (less memory, ~25%% slower)')
    parser.add_argument('-compile', action='store_true', help='torch.compile the model')
    parser.add_argument('-feature_size', type=int, default=24)
    parser.add_argument('-flash_attention', action='store_true', help='fused attention in SwinUNETR (MONAI >= 1.5)')
    # --- data ---
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-spatial_aug', action='store_true',
                        help='original random flips/rot90 (note: L/R flip changes the montage)')
    # --- validation / logging ---
    parser.add_argument('-val_every', type=int, default=2, help='validate every N epochs')
    parser.add_argument('-sw_batch_size', type=int, default=2)
    parser.add_argument('-overlap', type=float, default=0.25)
    parser.add_argument('-log_every', type=int, default=20)
    parser.add_argument('-resume', type=str, default=None, help='path to a last-checkpoint to resume from')
    parser.add_argument('-limit_batches', type=int, default=None, help='debug: batches per epoch')
    parser.add_argument('-seed', type=int, default=0)
    return parser.parse_args()


def validation(model, val_loader, inferer, device, autocast, limit=None, distributed=False):
    model.eval()
    total_mae = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    with torch.no_grad(), autocast():
        for i, batch in enumerate(val_loader):
            if limit is not None and i >= limit:
                break
            input_data = batch['input_data'].to(device, non_blocking=True).float()
            label = batch['label'].to(device, non_blocking=True)
            vector = batch['vector'].to(device, non_blocking=True)
            # Full-volume sliding window; only backbone + head (montage classifier not needed here).
            output = inferer(inputs=input_data, network=lambda x: model.predict(x, vector))
            total_mae += torch.abs(output.float() - label).mean()
            n += 1
    if distributed:  # each rank evaluated a disjoint shard of the validation set
        dist.all_reduce(total_mae)
        dist.all_reduce(n)
    return (total_mae / n.clamp(min=1)).item()


def main():
    args = get_args()
    # Multi-GPU: launch with torchrun (sets WORLD_SIZE / RANK / LOCAL_RANK). Single GPU: plain `python train.py`.
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    rank, local_rank = int(os.environ.get('RANK', 0)), int(os.environ.get('LOCAL_RANK', 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    is_main = rank == 0
    log = (lambda *a, **k: print(*a, **k, flush=True)) if is_main else (lambda *a, **k: None)
    torch.manual_seed(args.seed + rank)  # different crops per rank; DDP broadcasts rank-0 weights anyway
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'none': None}[args.amp]
    autocast = lambda: torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    scaler = torch.amp.GradScaler(enabled=args.amp == 'fp16')

    log('Loading Data...')
    train_loader, val_loader = create_dataloaders(args.batch_size, args.root_dir, args.x, args.y, args.z,
                                                  num_workers=args.num_workers, spatial_aug=args.spatial_aug,
                                                  rank=rank, world_size=world_size)
    log(f'  train samples: {len(train_loader.dataset)}  val samples/rank: {len(val_loader.dataset)}  '
        f'GPUs: {world_size}  global batch: {args.batch_size * world_size}')

    log('Define Model...')
    model = MultiModel(args.in_channels_img, args.in_channels_vector, args.out_channels, args.x, args.y, args.z,
                       feature_size=args.feature_size, use_checkpoint=args.use_checkpoint,
                       use_flash_attention=args.flash_attention).to(device)
    log(f'  parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    inferer = SlidingWindowInferer(roi_size=(args.x, args.y, args.z), sw_batch_size=args.sw_batch_size,
                                   overlap=args.overlap, mode='gaussian')

    criterion1 = nn.L1Loss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = args.limit_batches or len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup = int(args.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / max(1, warmup) if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    last_path = os.path.splitext(args.save_path)[0] + '_last.pth'
    start_epoch, best_mae, best_epoch, it_num = 0, float('inf'), -1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch, best_mae, best_epoch, it_num = ckpt['epoch'] + 1, ckpt['best_mae'], ckpt['best_epoch'], ckpt['it']
        log(f'Resumed from {args.resume} at epoch {start_epoch}')

    train_model = DDP(model, device_ids=[local_rank]) if distributed else model
    if args.compile:
        train_model = torch.compile(train_model)

    log('Begin Training...')
    for epoch in range(start_epoch, args.num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        train_model.train()
        sums = torch.zeros(4, device=device)  # loss1, loss2, loss3, total - accumulated on GPU (no per-step sync)
        n_log, t_data, t_epoch = 0, 0.0, time.time()
        t_last = time.time()
        for i, batch in enumerate(train_loader):
            if args.limit_batches is not None and i >= args.limit_batches:
                break
            input_data = batch['input_data'].to(device, non_blocking=True).float()
            label = batch['label'].to(device, non_blocking=True)
            vector = batch['vector'].to(device, non_blocking=True)
            vector_label = batch['montage'].to(device, non_blocking=True)
            t_data += time.time() - t_last

            with autocast():
                output, output_class = train_model(input_data, vector)
            output = output.float()
            loss1 = criterion1(output, label)
            loss2 = criterion2(output_class.float(), vector_label)
            loss3 = reg_ece(output, label)
            loss = loss1 + args.w_class * loss2 + args.w_ece * loss3

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            sched.step()

            sums += torch.stack([loss1.detach(), loss2.detach(), loss3.detach(), loss.detach()])
            n_log += 1
            it_num += 1
            if it_num % args.log_every == 0:
                l1, l2, l3, lt = (sums / n_log).tolist()
                dt = time.time() - t_epoch
                log(f'Epoch {epoch + 1} iter {i + 1}/{steps_per_epoch} (global {it_num}) | '
                      f'L1 {l1:.4f} CE {l2:.4f} ECE {l3:.4f} total {lt:.4f} | lr {sched.get_last_lr()[0]:.2e} | '
                      f'{dt / (i + 1):.2f}s/it (data wait {t_data / (i + 1):.2f}s/it)')
                sums.zero_()
                n_log = 0
            t_last = time.time()

        msg = f'Epoch [{epoch + 1}/{args.num_epochs}] done in {time.time() - t_epoch:.0f}s'
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.num_epochs:
            t_val = time.time()
            avg_mae = validation(model, val_loader, inferer, device, autocast, limit=args.limit_batches,
                                 distributed=distributed)
            msg += f', val MAE: {avg_mae:.5f} ({time.time() - t_val:.0f}s)'
            if avg_mae < best_mae:
                best_mae, best_epoch = avg_mae, epoch
                if is_main:
                    torch.save(model.state_dict(), args.save_path)
                msg += ' -> saved best'
        log(msg)

        if is_main:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': sched.state_dict(),
                        'scaler': scaler.state_dict(), 'epoch': epoch, 'best_mae': best_mae, 'best_epoch': best_epoch,
                        'it': it_num, 'args': vars(args)}, last_path)

    log(f'Best model achieved at Epoch {best_epoch + 1}, Best MAE: {best_mae}')
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
