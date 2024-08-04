from argparse import Namespace
from pathlib import Path
import toml
from dataset_utils import create_dataset, get_embeddings
from models import load_model
import torch
import numpy as np
from DiffJPEG.compression import compress_jpeg
from DiffJPEG.decompression import decompress_jpeg
from DiffJPEG.jpeg_utils import diff_round, quality_to_factor


criterion = torch.nn.functional.cosine_similarity
EMBEDDING_FNM = 'outputs/embeddings/{}_{}_embeddings.npy'

IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)
THERMAL_MEAN=(0.2989 * IMG_MEAN[0]) + (0.5870 * IMG_MEAN[1]) + (0.1140 * IMG_MEAN[2])
THERMAL_STD=(0.2989 * IMG_STD[0]) + (0.5870 * IMG_STD[1]) + (0.1140 * IMG_STD[2])

def unnorm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() * std) + mean).to(device)

def norm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() - mean) / std).to(device)


def unnorm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() * s) + m).to(device)

def norm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() - m) / s).to(device)

def threshold(X, eps, modality, device):
    if modality == 'vision':
        X_unnorm = unnorm(X.data)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1)), norm(torch.clamp(X_unnorm-eps, min=0, max=1))
    elif modality == 'thermal':
        X_max, X_min = torch.clamp(X+eps, min=0, max=1), torch.clamp(X-eps, min=0, max=1)
    elif modality == 'audio':
        X_max, X_min = X + eps, X - eps
    return X_max.to(device), X_min.to(device)

def extract_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    Path(cfg_dict['output_dir']).mkdir(parents=True, exist_ok=True)
    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    cfg_dict['jpeg'] = ('jpeg' in cfg_dict) and cfg_dict['jpeg']
    assert (not cfg_dict['jpeg']) or (cfg_dict['modality'] == 'vision')

    if cfg_dict['modality'] == 'vision':
        cfg_dict['epsilon'] = cfg_dict['epsilon'] / 255

    if type(cfg_dict['epochs']) == list:
        cfg_dict['max_epochs'] = max(cfg_dict['epochs'])
    else:
        cfg_dict['max_epochs'] = cfg_dict['epochs']
        cfg_dict['epochs'] = [cfg_dict['epochs']]

    assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def extract_eval_args(exp_name):
    fnm = f'configs/{exp_name}.toml'
    print(f'Loading config from {fnm}...')

    cfg_dict = toml.load(fnm)['general']

    if 'model_flag' in cfg_dict:
        cfg_dict['model_flags'] = [cfg_dict['model_flag']]
        cfg_dict['target_model_flag'] = cfg_dict['model_flag']
    cfg_dict['target_model_flag'] = cfg_dict.get('target_model_flag', None)

    if 'gpu_num' in cfg_dict:
        cfg_dict['gpu_nums'] = [cfg_dict['gpu_num']]

    assert cfg_dict['eval_type'] in ['adversarial', 'organic', 'transfer']
    if cfg_dict['eval_type'] == 'transfer':
        assert 'adv_file' in cfg_dict

    assert cfg_dict['number_images'] % cfg_dict['batch_size'] == 0
    return Namespace(**cfg_dict)

def jpeg(x, height=224, width=224, rounding=diff_round, quality=80):
    img_tensor = unnorm(x).squeeze(0)
    factor = quality_to_factor(quality)
    y, cb, cr = compress_jpeg(img_tensor, rounding=rounding, factor=factor)
    img_tensor = decompress_jpeg(y, cb, cr, height, width, rounding=rounding, factor=factor)
    return norm(img_tensor)

def pgd_step(model, X, Y, X_min, X_max, lr, modality, device):
    embeds = model.forward(X, modality, normalize=False)
    loss = 1 - criterion(embeds, Y, dim=1)
    update = lr * torch.autograd.grad(outputs=loss.mean(), inputs=X)[0].sign()
    X = (X.detach().cpu() - update.detach().cpu()).to(device)
    X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
    return X, embeds, loss.clone().detach().cpu()

def gpu_num_to_device(gpu_num):
    return f'cuda:{gpu_num}' if torch.cuda.is_available() and gpu_num >= 0 else 'cpu'

def load_model_data_and_dataset(dataset_flag, model_flags, gpus, seed):
    devices = [gpu_num_to_device(g) for g in gpus]  
    models = [load_model(f, devices[i % len(devices)]) for i, f in enumerate(model_flags)]
    dataset = create_dataset(dataset_flag, model=models[0], device=devices[0], seed=seed)
    embeddings = []
    for i, f in enumerate(model_flags):
        fnm = EMBEDDING_FNM.format(dataset_flag, f)
        embeddings.append(get_embeddings(fnm, dataset.label_texts, devices[i % len(devices)],
                                         dataset_flag, models[i]))
    model_data = [
        (m, e, devices[i % len(devices)]) for i, (m, e) in enumerate(zip(models, embeddings))
    ]
    return model_data, dataset

def print_results(ranks, losses, model_flag):
    if type(ranks) == list:
        ranks = np.concatenate(ranks)
    if type(losses) == list:
        losses = np.concatenate(losses)

    top1 = f'{(ranks < 1).mean():.2f}'
    top5 = f'{(ranks < 5).mean():.2f}'
    mean = f'{np.mean(losses):.4f}'
    stddev = f'{np.std(losses):.4f}'
    print(f'{model_flag},{top1},{top5},{mean},{stddev}')
