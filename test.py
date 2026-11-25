#Load Standard Libraries
import argparse
import os
from collections import defaultdict

import nibabel as nib
import numpy as np

#Load Torch Libraries
import torch

#Load Monai Packages
from monai.inferers import SlidingWindowInferer
from monai.metrics import SSIMMetric

#Load Custom
from utils import create_test_dataloader, MultiModel, reg_ece


def center_crop(t, roi):
    sl = [slice(None), slice(None)]
    for dim, r in zip(t.shape[2:], roi):
        s = max(0, (dim - r) // 2)
        sl.append(slice(s, s + r))
    return t[tuple(sl)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=int, default=256)
    parser.add_argument('-y', type=int, default=256)
    parser.add_argument('-z', type=int, default=256)
    parser.add_argument('-root_dir', type=str, default='cache', help='cache directory written by preprocess.py')
    parser.add_argument('-in_channels_img', type=int, default=1)
    parser.add_argument('-in_channels_vector', type=int, default=2)
    parser.add_argument('-out_channels', type=int, default=1)
    parser.add_argument('-feature_size', type=int, default=24)
    parser.add_argument('-flash_attention', action='store_true')
    parser.add_argument('-save_path', type=str, default='Models/best_model.pth', help='trained model (state_dict)')
    parser.add_argument('-results_path', type=str, default='best_model', help='sub-folder of Results/')
    parser.add_argument('-sw_batch_size', type=int, default=2)
    parser.add_argument('-overlap', type=float, default=0.25)
    parser.add_argument('-label_clip', type=float, default=3.0, help='clip labels above this (as in original test)')
    parser.add_argument('-amp', choices=['bf16', 'fp16', 'none'], default='bf16')
    parser.add_argument('-no_save', action='store_true', help='do not write prediction NIfTIs')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'none': None}[args.amp]
    roi = (args.x, args.y, args.z)

    print('Loading Data...', flush=True)
    test_loader = create_test_dataloader(args.root_dir)

    print('Define Model...', flush=True)
    model = MultiModel(args.in_channels_img, args.in_channels_vector, args.out_channels, *roi,
                       feature_size=args.feature_size, use_flash_attention=args.flash_attention).to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.eval()

    inferer = SlidingWindowInferer(roi_size=roi, sw_batch_size=args.sw_batch_size, overlap=args.overlap,
                                   mode='gaussian')
    ssim = SSIMMetric(spatial_dims=3)
    results_dir = os.path.join('Results', args.results_path)
    os.makedirs(results_dir, exist_ok=True)

    # group 'old' = 256^3 in-house scans, 'young' = 260x311x260 HCP scans (same split as original script)
    metrics = defaultdict(lambda: defaultdict(float))
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for batch in test_loader:
            input_data = batch['input_data'].to(device).float()
            vector = batch['vector'].to(device)
            label = batch['label'].to(device).clamp(max=args.label_clip)
            vector_label = batch['montage'].to(device)
            name = batch['Name'][0]

            output = inferer(inputs=input_data, network=lambda x: model.predict(x, vector)).float()
            output_class = model(center_crop(input_data, roi), vector)[1].argmax(dim=1)

            m = {
                'MAE': torch.abs(output - label).mean().item(),
                'MSE': ((output - label) ** 2).mean().item(),
                'SSIM': ssim(output, label).mean().item(),
                'ECE': reg_ece(output, label).item(),
                'Accuracy': 100.0 * (output_class == vector_label).float().item(),
            }
            group = 'Old' if batch['shape'][0][1].item() == 256 else 'Young'
            for g in ('All', group):
                metrics[g]['n'] += 1
                for k, v in m.items():
                    metrics[g][k] += v

            if not args.no_save:
                affine = np.load(os.path.join(args.root_dir, 'Test', f"{batch['subject'][0]}_affine.npy"))
                nib.save(nib.Nifti1Image(output.squeeze().cpu().numpy(), affine),
                         os.path.join(results_dir, f'{name}_output_J.nii.gz'))
            print(f'Finished Running: {name}, ' + ', '.join(f'{k}: {v:.5f}' for k, v in m.items()), flush=True)

    for g, d in metrics.items():
        n = d.pop('n')
        print(f'{g} (n={int(n)}) - ' + ', '.join(f'{k}: {v / n:.5f}' for k, v in d.items()), flush=True)


if __name__ == '__main__':
    main()
