import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import load_model
from utils import unnorm, norm
import imagebind.data as data
from torchvision.utils import save_image
import torch
import torchvision.transforms as transforms


device="cuda:0"
modality="vision"
batch_size = 1
seed=0
criterion = F.cosine_similarity

IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)

def unnorm(tensor, mean=IMG_MEAN, std=IMG_STD):
    m = torch.tensor(IMG_MEAN)[None, :, None, None].to(device)
    s = torch.tensor(IMG_STD)[None, :, None, None].to(device)
    return (tensor.clone().to(device) * s) + m

def norm(tensor, mean=IMG_MEAN, std=IMG_STD):
    m = torch.tensor(IMG_MEAN)[None, :, None, None].to(device)
    s = torch.tensor(IMG_STD)[None, :, None, None].to(device)
    return (tensor.clone().to(device) - m) / s

transform = transforms.ToPILImage()
criterion = F.cosine_similarity

def jpeg_transform(X):
    image_filename = 'dummy.jpg'
    save_image(torch.squeeze(unnorm(X)), image_filename)
    jpeg_X = data.load_and_transform_vision_data([image_filename], device)
    os.remove(image_filename)
    jpeg_X=jpeg_X.clamp(0,1)
    return jpeg_X.squeeze().cpu()

# Define a list of transformations to apply
transformations = [
    ("JPEG", jpeg_transform),
    ("GaussianBlur", transforms.GaussianBlur(kernel_size=9, sigma=(1.0, 5.0))),
    ("RandomAffine", transforms.RandomAffine(degrees=45)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)),
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1)),
    ("RandomPerspective", transforms.RandomPerspective(distortion_scale=0.5, p=0.5))
]

def analyze_similarity(file_path, model):
    X_advs = np.load(file_path)
    X_advs = torch.tensor(X_advs).to(device)
    similarity_transform = [[] for _ in range(len(transformations))]
    for X in X_advs:
        X = X.unsqueeze(0)
        with torch.no_grad():
            embeds = model.forward(X, modality, normalize=False)
            for j, (name, transform) in enumerate(transformations):
                transformed_image = norm(transform(unnorm(X).detach().cpu())) if callable(transform) else norm(transform(unnorm(X).detach().cpu()))
                transformed_embeds = model.forward(transformed_image, modality, normalize=False)
                similarity_transform[j].append(criterion(embeds[:, None, :], transformed_embeds[:, None, :], dim=2)[0][0].item())
    
    for (name, _), sims in zip(transformations, similarity_transform):
        sims = np.array(sims)
        mean_sim = sims.mean()
        std_sim = sims.std()
        print(f"    {name}: Mean = {mean_sim}, Std Dev = {std_sim}")

print('-----------Imagebind-----------')
model = load_model("imagebind", device)
file_path = 'outputs/imagenet/whitebox/imagebind/x_inits.npy'
print("X_inits:")
analyze_similarity(file_path, model)
print("X_advs:")
file_path = 'outputs/imagenet/whitebox/imagebind/x_advs_300.npy'
analyze_similarity(file_path, model)
print("X_advs_jpeg:")
file_path = 'outputs/imagenet/whitebox/imagebind_jpeg/x_advs_300.npy'
analyze_similarity(file_path, model)

print('-----------Audioclip-----------')
model = load_model("audioclip", device)
file_path = 'outputs/imagenet/whitebox/audioclip/x_inits.npy'
print("X_inits:")
analyze_similarity(file_path, model)
print("X_advs:")
file_path = 'outputs/imagenet/whitebox/audioclip/x_advs_300.npy'
analyze_similarity(file_path, model)
print("X_advs_jpeg:")
file_path = 'outputs/imagenet/whitebox/audioclip_jpeg/x_advs_300.npy'
analyze_similarity(file_path, model)
