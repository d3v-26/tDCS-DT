#Load Standard Libraries
import inspect
import os

import numpy as np

#Load Torch Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset

#Load Monai Packages
from monai.networks.nets import SwinUNETR, EfficientNet

##########################################################################################################################

# Dataloading Tools
#
# Data comes from the .npy cache written by preprocess.py (T1 already scaled to [0, 1]).
# Each (subject, montage) pair is one sample, exactly like the original CustomDataset:
#   vector [[1],[0]] -> F3-F4,  vector [[0],[1]] -> C3-Fp2

MONTAGES = ('F3F4', 'C3Fp2')


def load_ids(cache_dir, split):
    path = os.path.join(cache_dir, split, 'ids.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - run preprocess.py first')
    return [str(s) for s in np.load(path)]


class TDCSDataset(Dataset):
    def __init__(self, cache_dir, split, roi_size=None, train=False, spatial_aug=False, ids=None):
        self.dir = os.path.join(cache_dir, split)
        self.ids = ids if ids is not None else load_ids(cache_dir, split)
        self.items = [(sid, m) for sid in self.ids for m in range(len(MONTAGES))]
        self.roi_size = tuple(roi_size) if roi_size is not None else None
        self.train = train
        self.spatial_aug = spatial_aug

    def __len__(self):
        return len(self.items)

    def _crop_slices(self, shape):
        # Random crop for training, centre crop otherwise (original val/test used RandSpatialCropd,
        # which made the validation metric non-deterministic).
        sl = []
        for dim, roi in zip(shape, self.roi_size):
            if dim <= roi:
                sl.append(slice(0, dim))
                continue
            start = np.random.randint(0, dim - roi + 1) if self.train else (dim - roi) // 2
            sl.append(slice(start, start + roi))
        return tuple(sl)

    def _pad(self, arr):
        pad = [(0, max(0, r - s)) for s, r in zip(arr.shape, self.roi_size)]
        return np.pad(arr, pad) if any(p[1] for p in pad) else arr

    def __getitem__(self, index):
        sid, m = self.items[index]
        t1 = np.load(os.path.join(self.dir, f'{sid}_T1.npy'), mmap_mode='r')
        label = np.load(os.path.join(self.dir, f'{sid}_J_{MONTAGES[m]}.npy'), mmap_mode='r')
        shape = t1.shape

        if self.roi_size is not None:
            sl = self._crop_slices(shape)
            t1, label = self._pad(np.asarray(t1[sl])), self._pad(np.asarray(label[sl]))
        else:
            t1, label = np.asarray(t1), np.asarray(label)

        if self.train and self.spatial_aug:
            # Same augmentation as the original pipeline (RandFlipd x3 p=0.1, RandRotate90d p=0.1).
            # NOTE: an L/R flip turns C3-Fp2 into C4-Fp1, so this mislabels data; off by default.
            for axis in range(3):
                if np.random.rand() < 0.1:
                    t1, label = np.flip(t1, axis), np.flip(label, axis)
            if np.random.rand() < 0.1:
                k = np.random.randint(1, 4)
                t1, label = np.rot90(t1, k, axes=(0, 1)), np.rot90(label, k, axes=(0, 1))

        vector = torch.zeros(len(MONTAGES), 1)
        vector[m, 0] = 1.0
        return {
            'input_data': torch.from_numpy(np.ascontiguousarray(t1)).unsqueeze(0),        # float16, [0,1]
            'label': torch.from_numpy(np.ascontiguousarray(label)).float().unsqueeze(0),
            'vector': vector,
            'montage': m,
            'Name': f'{sid}_{MONTAGES[m]}',
            'subject': sid,
            'shape': torch.tensor(shape),
        }


def _loader(dataset, batch_size, shuffle, num_workers, drop_last=False, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)


def create_dataloaders(batch_size, cache_dir, x, y, z, num_workers=8, spatial_aug=False, rank=0, world_size=1):
    """batch_size is per GPU. With world_size > 1 the training set is sharded with a DistributedSampler and the
    validation set is split round-robin across ranks (each rank evaluates a disjoint subset)."""
    train_ds = TDCSDataset(cache_dir, 'Train', roi_size=(x, y, z), train=True, spatial_aug=spatial_aug)
    # Validation runs on the full volume with a sliding window (shapes differ -> batch size 1).
    val_ds = TDCSDataset(cache_dir, 'Val', roi_size=None, train=False)
    if world_size > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = _loader(train_ds, batch_size, False, num_workers, drop_last=True, sampler=sampler)
        val_ds = Subset(val_ds, list(range(rank, len(val_ds), world_size)))
    else:
        train_loader = _loader(train_ds, batch_size, True, num_workers, drop_last=True)
    val_loader = _loader(val_ds, 1, False, max(1, num_workers // 2))
    return train_loader, val_loader


def create_test_dataloader(cache_dir, num_workers=4):
    return _loader(TDCSDataset(cache_dir, 'Test', roi_size=None, train=False), 1, False, num_workers)

##########################################################################################################################

# Model


class MultiModel(nn.Module):
    def __init__(self, in_channels_img, in_channels_vector, out_channels, x, y, z,
                 feature_size=24, use_checkpoint=False, use_flash_attention=False):
        super(MultiModel, self).__init__()
        kw = dict(in_channels=in_channels_img, out_channels=out_channels, spatial_dims=3,
                  feature_size=feature_size, use_checkpoint=use_checkpoint)
        if 'img_size' in inspect.signature(SwinUNETR.__init__).parameters:  # removed in MONAI >= 1.5
            kw['img_size'] = (x, y, z)
        if use_flash_attention:  # MONAI >= 1.5: fused SDPA, avoids materialising the window-attention matrix
            if 'use_flash_attention' not in inspect.signature(SwinUNETR.__init__).parameters:
                raise ValueError('use_flash_attention needs MONAI >= 1.5')
            kw['use_flash_attention'] = True
        self.backbone = SwinUNETR(**kw)

        self.head = nn.Conv3d(out_channels + in_channels_vector, 1, kernel_size=1)

        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r4_k5_s22_e6_i192_o320_se0.25",
        ]
        self.montage = EfficientNet(blocks_args_str, num_classes=2, image_size=x, spatial_dims=3, in_channels=1)

    def predict(self, x_img, x_vector):
        """E-field map only (no montage classifier) - used for sliding-window inference."""
        backbone_output = self.backbone(x_img)
        # x_vector: B x 2 x 1 -> B x 2 x 1 x 1 x 1, broadcast over space (expand: no memory copy).
        # A batch-1 vector is also broadcast over the batch (sliding-window inference with sw_batch_size > 1).
        b = backbone_output.shape[0]
        x_vector = x_vector.to(backbone_output.dtype)[..., None, None].expand(b, -1, *backbone_output.shape[2:])
        return self.head(torch.cat([backbone_output, x_vector], dim=1))

    def forward(self, x_img, x_vector):
        output = self.predict(x_img, x_vector)
        output_class = self.montage(output)
        return output, output_class

##########################################################################################################################

# Other


def reg_ece(predictions, targets, num_bins=10, device=None):
    """Regression calibration error: mean |pred - target| per prediction-value bin, averaged over bins.

    Torch re-implementation of the original numpy version. The numpy version could not run on CUDA
    tensors that require grad and, even on CPU, produced a constant (no gradient). Here bin membership
    is treated as constant and the gradient flows through the per-bin errors. Empty bins are ignored
    (the original counted them as 0 and silently dropped the last bin).
    """
    pred = predictions.float().flatten()
    err = (pred - targets.float().flatten()).abs()
    with torch.no_grad():
        lo, hi = pred.min(), pred.max()
        edges = torch.linspace(0, 1, num_bins + 1, device=pred.device) * (hi - lo) + lo
        idx = torch.bucketize(pred, edges[1:-1], right=True)
        counts = torch.bincount(idx, minlength=num_bins)
    sums = torch.zeros(num_bins, device=pred.device, dtype=err.dtype).scatter_add(0, idx, err)
    nonempty = counts > 0
    return (sums[nonempty] / counts[nonempty]).mean()


def vector_class(vector, device='cpu'):
    """[[1],[0]] -> 0 (F3F4), [[0],[1]] -> 1 (C3Fp2)."""
    return vector[:, :, 0].argmax(dim=1).to(device)
