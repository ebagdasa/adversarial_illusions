import os
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import librosa
from tqdm import tqdm
import imagebind.data as data


DATA_PATH = {
    'imagenet': 'data/imagenet/',
    'audiocaps': 'data/AudioCaps/',
    'audioset': 'data/AudioCaps/',
}

TEMPLATES = {
    'imagenet': 'A photo of a {}.'
}


def get_embeddings(embs_file, labels, device, dataset_flag, model=None, batch_size=250, device_override=False):
    if embs_file is not None and os.path.isfile(embs_file):
        print(f'Reading label embeddings from {embs_file}...')
        return torch.tensor(np.load(embs_file)).to(device)
    
    print(f'No label embeddings found. Generating...')
    if dataset_flag in TEMPLATES:
        labs = np.stack([TEMPLATES[dataset_flag].format(labels[i].split(',')[0]) for i in range(len(labels))])
    else:
        labs = labels
    embs = []

    for i in tqdm(range(int(np.ceil((len(labs) / batch_size))))):
        batch = labs[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            embs.append(model.cpu().forward(batch, 'text', normalize=False))

    if not device_override:
        model.to(device)

    if embs_file is not None:
        print(f'Writing label embeddings to {embs_file}...')
        print()
        folder_path = os.path.dirname(embs_file)
        os.makedirs(folder_path, exist_ok=True)
        np.save(embs_file, torch.cat(embs).to(device).cpu())
    return torch.concatenate(embs).to(device)


class WrappedImageNetDataset(Dataset):
    def __init__(
        self, dataset, labels, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        self.seed = seed
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(dataset))
        self.device = device
        self.embs_file = embs_input
        self.label_texts = labels
        if self.embs_file is not None:
            self.labels = get_embeddings(self.embs_file, self.label_texts, self.device, 'imagenet',
                                         self.model, embedding_batch_size, embedding_override)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig_id = self.dataset[idx]
        gt, y_id = self.dataset[self.mapping[idx]]
        if self.embs_file is not None:
            y = self.labels[y_id].to(self.device)
            return torch.squeeze(x), torch.squeeze(y), torch.squeeze(gt), y_id, y_orig_id
        return torch.squeeze(x), torch.squeeze(gt), y_id, y_orig_id


class WrappedAudioCapsDataset(Dataset):
    def __init__(
        self, dataset, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        self.seed = seed
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(dataset))
        self.device = device
        self.embs_file = embs_input
        self.label_texts = list(dict.fromkeys([y for _, y in dataset]))
        if self.embs_file is not None:
            self.labels = get_embeddings(self.embs_file, self.label_texts, self.device, 'AudioCaps',
                                         self.model, embedding_batch_size, embedding_override)
        self.lab_to_id = {l: i for i, l in enumerate(self.label_texts)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig = self.dataset[idx]
        gt, y_str = self.dataset[self.mapping[idx]]
        y_orig_id, y_str_id = self.lab_to_id[y_orig], self.lab_to_id[y_str]
        if self.model.flag == 'imagebind':
            x = torch.squeeze(x)[:, None, :, :]
            gt = torch.squeeze(gt)[:, None, :, :]
        x = (0.0001 * torch.randn_like(x)) + x.detach()
        if self.embs_file is not None:
            y = self.labels[y_str_id].to(self.device)
            return torch.squeeze(x), torch.squeeze(y), gt, y_str_id, y_orig_id
        return x, gt, y_str_id, y_orig_id


class AudioDataset(Dataset):
    def __init__(self, audio_dir, split_file, extension='wav', device='cpu', model_flag='imagebind'):
        self.audio_files = glob.glob(f'{audio_dir}*.{extension}')
        self.split = pd.read_csv(split_file, index_col='youtube_id')[['caption']]
        self.device = device
        self.model_flag = model_flag
        assert len(self.audio_files) > 0
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        path = self.audio_files[idx]
        if self.model_flag == 'imagebind':
            X = data.load_and_transform_audio_data([path], self.device)
        elif self.model_flag == 'audioclip':
            X = librosa.load(path, sr=44100, dtype=np.float32)[0]
            X = torch.tensor(X).to(self.device)
        y = self.split.loc[self.get_id(path)].iloc[-1].item()
        return X, y

    def get_id(self, path):
        return path.split('/')[-1].split('.')[0]


def imagenet_loader(path, model, device='cpu'):
    if model.flag == 'imagebind' or model.flag == 'audioclip':
        return data.load_and_transform_vision_data([path], device)
    elif model.flag == 'openclip':
        image_outputs = []
        with open(path, 'rb') as fopen:
            image = Image.open(fopen).convert('RGB')

        image = model.preprocess(image).to(device)
        image_outputs.append(image)
        return torch.stack(image_outputs, dim=0)
    else:
        raise NotImplementedError()


def create_dataset(
    dataset_flag, model=None, mapping=None, device='cpu', embs_input=None, seed=0
):
    assert model is not None
    if dataset_flag == 'imagenet':
        loader = lambda p: imagenet_loader(p, model, device)
        imagenet = datasets.ImageNet(DATA_PATH[dataset_flag], split='val', loader=loader)
        with open(DATA_PATH[dataset_flag] + 'imagenet1000_clsidx_to_labels.txt') as f:
            labels = eval(f.read().replace('\n', ''))
        return WrappedImageNetDataset(imagenet, labels, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audiocaps':
        audiocaps = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                                 DATA_PATH[dataset_flag] + 'splits/retrieval_test.csv',
                                 'wav',
                                 model_flag=model.flag)
        return WrappedAudioCapsDataset(audiocaps, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audioset':
        audioset = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                                DATA_PATH[dataset_flag] + 'splits/classification_test.csv',
                                'wav',
                                model_flag=model.flag)
        return WrappedAudioCapsDataset(audioset, model, mapping, device, seed, embs_input)
    else:
        raise NotImplementedError
