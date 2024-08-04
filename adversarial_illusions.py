import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import extract_args, jpeg, pgd_step, print_results, threshold,\
                  load_model_data_and_dataset, criterion


# Validate and configure experiment
cfg = extract_args(sys.argv[1])

# Instantiate models, devices, and datasets
print('Loading models and data...')
model_data, dataset = load_model_data_and_dataset(cfg.dataset_flag, cfg.model_flags,
                                                  cfg.gpu_nums, cfg.seed)
if cfg.target_model_flag is not None:
    if cfg.target_model_flag in cfg.model_flags:
        print('Using (one of) the same model(s) for target and input...')
        target_tup = model_data[cfg.model_flags.index(cfg.target_model_flag)]
    else:
        target_tup, _ =\
            load_model_data_and_dataset(cfg.dataset_flag, [cfg.target_model_flag],
                                        [cfg.target_gpu_num], cfg.seed)[0]

data_device = model_data[0][2]              # Assign initial device to hold data
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Create Empty Lists for Logging
X_advs = {e: [] for e in cfg.epochs}
X_inits, gts = [], []                       # Initial images and ground truths
adv_loss, gt_loss = [], []                  # Ground truth and adversarial distances
classified, ranks = [], []                  # Classification and rank information
y_ids, y_origs = [], []                     # Target and input label ids

# Create Adversarial Examples
print('Generating Illusions...')
torch.manual_seed(cfg.seed)
for i, (X, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (cfg.number_images // cfg.batch_size):
        break
    X_init = X.clone().detach().cpu()
    X_max, X_min = threshold(X, cfg.epsilon, cfg.modality, data_device)

    pbar = tqdm(range(cfg.max_epochs))
    lr = cfg.lr
    for j in pbar:
        X = jpeg(X.cpu()).to(data_device) if cfg.jpeg else X
        total_loss = torch.tensor([0.0] * cfg.batch_size)
        for m, l, d in model_data:
            Y = l[y_id]
            X_m, Y_m = X.to(d).requires_grad_(True), Y.to(d)
            X, embeds, loss = pgd_step(m, X_m, Y_m, X_min, X_max, lr, cfg.modality, data_device)
            total_loss += loss.clone().detach().cpu()
        pbar.set_postfix({'loss': total_loss / len(model_data), 'lr': lr})

        if j + 1 in cfg.epochs:
            X_advs[j+1].append(X.detach().cpu().clone())

        if ((j + 1) % cfg.gamma_epochs) == 0:
            lr *= 0.9
    
    # Record batchwise information
    if (cfg.target_model_flag is not None) and (target_tup[2] is not None):
        embeds = target_tup[0].forward(X.to(target_tup[2]), cfg.modality, normalize=False).detach().cpu()
        gt_embeddings = target_tup[0].forward(gt.to(target_tup[2]), cfg.modality, normalize=False).detach().cpu()
        classes = criterion(embeds[:, None, :], target_tup[1][None, :, :].cpu(), dim=2).argsort(dim=1, descending=True)
        adv_loss.append(criterion(embeds.detach().cpu(), target_tup[1][y_id].cpu(), dim=1))
        gt_loss.append(criterion(gt_embeddings, target_tup[1][y_id].cpu(), dim=1))
        classified.append((classes == y_id[:, None])[:, 0].cpu())
        ranks.append((classes == y_id[:, None]).int().argmax(axis=1))
        np.save(cfg.output_dir + 'adv_loss', np.concatenate(adv_loss))
        np.save(cfg.output_dir + 'gt_loss', np.concatenate(gt_loss))
        np.save(cfg.output_dir + 'classified', np.concatenate(classified))
        np.save(cfg.output_dir + 'ranks', np.concatenate(ranks))


    if cfg.modality == 'vision':
        X_inits.append(X_init.clone())
        gts.append(gt.cpu().clone())
        y_origs.append(y_orig.cpu())
        y_ids.append(y_id.cpu())

        np.save(cfg.output_dir + 'x_inits', np.concatenate(X_inits))
        np.save(cfg.output_dir + 'gts', np.concatenate(gts))
        np.save(cfg.output_dir + 'y_origs', np.concatenate(y_origs))
        np.save(cfg.output_dir + 'y_ids', np.concatenate(y_ids))

        for k, v in X_advs.items():
            np.save(cfg.output_dir + f'x_advs_{k}', np.concatenate(X_advs[k]))

print('Training Complete...')
if (cfg.target_model_flag is not None) and (target_tup[2] is not None):
    print("Final Results...")
    print()
    print(f'{cfg.dataset_flag},Top-1,Top-5,mean,std')
    print_results(ranks, adv_loss, cfg.target_model_flag)
