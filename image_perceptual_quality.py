from zipfile import Path

import pyiqa
import torch
import numpy as np
IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)


def evaluate_images(output_path, metrics, advs_path="x_advs.npy"):
    x_advs = normalize_01(torch.from_numpy(np.load(output_path + f"/{advs_path}")))
    x_init = normalize_01(torch.from_numpy(np.load(output_path + "/x_inits.npy")))
    
    sums = {k: 0.0 for k in metrics}
    n = 0
    batch_size = x_advs.shape[0]
    print(f"Evaluating {output_path}...")
    for name, metric in metrics.items():
        print(f"Evaluating {name}...")
        for i in range(batch_size):
            x_adv_i = x_advs[i:i + 1]  # shape: (1, C, W, H)
            x_init_i = x_init[i:i + 1]  # shape: (1, C, W, H)
            score = metric(x_adv_i, x_init_i)
            score_val = float(score.mean().detach().cpu())
            sums[name] += score_val
            n += 1
    n = n/len(metrics.keys())  # Average over metrics
    means = {name: (sums[name] / n if n > 0 else float("nan")) for name in sums}
    return means

def normalize_01(x: torch.Tensor) -> torch.Tensor:
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


if __name__ == "__main__": 
    paths = [
        'outputs/imagenet/query/audioclip-temp-2_0',
        'outputs/imagenet/query/audioclip-temp-2_0',
        'outputs/imagenet/query/audioclip-temp-1_0',
        'outputs/imagenet/query/audioclip-temp-0_7',
        'outputs/imagenet/query/audioclip-temp-0_5',
        'outputs/imagenet/query/audioclip-temp-0_4',
        'outputs/imagenet/query/audioclip-temp-0_3',
        'outputs/imagenet/query/audioclip-temp-0_1',
        ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    metrics = {
            "PSNR": pyiqa.create_metric("psnr", device=device),
            "LPIPS": pyiqa.create_metric("lpips", device=device),
            "MS-SSIM": pyiqa.create_metric("ms_ssim", device=device),
            "AHIQ": pyiqa.create_metric("ahiq", device=device),
        }
    means = {}
    for output_path in paths:
        if 'whitebox' in output_path:
            advs_paths = ["x_advs_300.npy", 'x_advs_7500.npy']
        else:
            advs_paths = ["x_advs.npy"]
            
        for advs_path in advs_paths:
            mean = evaluate_images(output_path, metrics, advs_path=advs_path)
            input_name = '-'.join(output_path.split('/')[-2:])  
            
            if advs_path != "x_advs.npy":
                input_name += f"-{advs_path.split('.')[0].split('_')[-1]}"
                
            print(f"Mean scores for {input_name}: {mean}")
            means[input_name] = mean
    print(f"Overall mean scores: {means}")
        