import torch.nn as nn

from imagebind.models import imagebind_model
import imagebind.data as data

from AudioCLIP import AudioCLIP
import open_clip
from open_clip import tokenizer


def load_model(model_flag, device):
    if model_flag == 'imagebind':
        model = ImageBindWrapper(imagebind_model.imagebind_huge(pretrained=True), device=device)
    elif model_flag == 'audioclip':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Full-Training.pt'))
    elif model_flag == 'audioclip_partial':
        model = AudioCLIPWrapper(AudioCLIP(pretrained=f'bpe/AudioCLIP-Partial-Training.pt'))
    elif model_flag == 'openclip':
        m, _, p = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif model_flag == 'openclip_rn50':
        m, _, p = open_clip.create_model_and_transforms('RN50', pretrained='openai', cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    elif 'openclip' in model_flag:
        _, backbone, pretrained = model_flag.split(';')
        m, _, p = open_clip.create_model_and_transforms(backbone, pretrained=pretrained, cache_dir='bpe/')
        model = OpenCLIPWrapper(m, p)
    else:
        raise NotImplementedError()

    model.to(device)
    model.eval()
    return model


class ImageBindWrapper(nn.Module):
    def __init__(self, model, device):
        super(ImageBindWrapper, self).__init__()
        self.model = model
        self.device = device
        self.flag = 'imagebind'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = data.load_and_transform_text(X, self.device)
            X = X.to(next(self.model.parameters()).device)
        return self.model.forward({modality: X}, normalize=normalize)[modality]


class AudioCLIPWrapper(nn.Module):
    def __init__(self, model):
        super(AudioCLIPWrapper, self).__init__()
        self.model = model
        self.flag = 'audioclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        if modality == 'text':
            if isinstance(X, str):
                X = [X]
            X = [[i] for i in X]

        features = self.model.forward(**{modality: X}, normalize=normalize)[0][0]
        if modality == 'audio':
            return features[0]
        elif modality == 'image':
            return features[1]
        elif modality == 'text':
            return features[2]
        else:
            raise NotImplementedError()
        

class OpenCLIPWrapper(nn.Module):
    def __init__(self, model, preprocess):
        super(OpenCLIPWrapper, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.flag = 'openclip'
    
    def forward(self, X, modality, normalize=True):
        if modality == 'vision':
            modality = 'image'
        elif modality == 'text':
            X = tokenizer.tokenize(X)

        features = self.model.forward(**{modality: X})
        if modality == 'image':
            return features[0]
        elif modality == 'text':
            return features[1]
        else:
            raise NotImplementedError()
