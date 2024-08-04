import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import criterion, extract_eval_args, load_model_data_and_dataset, print_results


cfg = extract_eval_args(sys.argv[1])

# Load Data
model_data, dataset = load_model_data_and_dataset(cfg.dataset_flag, cfg.model_flags,
                                                  cfg.gpu_nums, cfg.seed)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

print(f'Results ({cfg.eval_type})...')
print()
print(f'{cfg.dataset_flag},Top-1,Top-5,mean,std')
for i, (model, labels, device) in enumerate(model_data):
    model.eval()
    with torch.no_grad():
        if cfg.eval_type == 'transfer':
            x_advs = torch.tensor(np.load(cfg.adv_file)).to(device)
        ranks = []
        losses = []

        torch.manual_seed(0)
        for i, (src, gt, y, _) in enumerate(dataloader):
            if i >= (cfg.number_images // cfg.batch_size):
                break
            if cfg.eval_type == 'transfer':
                x = (x_advs[cfg.batch_size * i: cfg.batch_size * (i + 1)])
            elif cfg.eval_type == 'adversarial':
                x = src.to(device)
            else:
                x = gt.to(device)

            y_emb = labels[y]
            embeds = model.forward(x, modality=cfg.modality).detach()
            classes = criterion(embeds[:, None, :], labels[None, :, :], dim=2).argsort(dim=1, descending=True)
            ranks.append((classes.cpu() == y[:, None]).int().argmax(axis=1))
            losses.append(criterion(embeds, y_emb, dim=1).cpu().detach().numpy())
        print_results(ranks, losses, model.flag)
