import os
from tqdm import tqdm
import toml
import numpy as np
import torch

import torch.optim as optim
from utils import threshold, criterion, unnorm
from dataset_utils import imagenet_loader
from models import load_model
from torchvision.utils import save_image


# Configure Script
config = toml.load('configs/imagenet/whitebox/imagebind.toml')['general']

gpu_num = config['gpu_num']
epochs = config['epochs']
batch_size = config['batch_size']
eps = config['epsilon']
zero_shot_steps = config['zero_shot_steps']
lr = config['lr']
eta_min = config['eta_min']
seed = config['seed']
output_dir = config['output_dir']
n_images = config['number_images']
buffer_size = config['buffer_size']
delta = config['delta']
model_flag = config.get('model_flag', 'imagebind')
embs_input = config.get('embeddings_input', output_dir + 'embs.npy')\
                   .format(model_flag)
gamma_epochs = config.get('gamma_epochs', 100)
modality = config.get('modality', 'vision')
dataset_flag = config.get('dataset_flag', 'imagenet')

if modality == 'vision':
    eps = eps / 255
max_epochs=2000
device = f"cuda:{gpu_num}" if torch.cuda.is_available() and gpu_num >= 0 else "cpu"
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(model_flag, device)

input_dir = 'data/thermal_dataset/raw_images/'
output_dir = 'outputs/thermal/perturbed_images_text_generation/'
os.makedirs(output_dir, exist_ok=True)


target_text=["criminal with a gun"]
Y = model.forward(target_text, "text", normalize=False).to(device)

for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
    if filename.endswith(".jpg"):  # Assuming all images are PNGs
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        output_path = output_path.replace(".jpg", ".png")

        # Load and prepare the image
        X = imagenet_loader(image_path, model, device).to(device).requires_grad_(True)
        X_init = X.clone().detach().cpu().requires_grad_(False)
        X = X.to(device).requires_grad_(True) 

        X_max, X_min = threshold(X, eps, modality, device)
        optimizer = optim.SGD([X], lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            np.arange(gamma_epochs, max_epochs, gamma_epochs),
                                            gamma=0.9)
        for j in range(max_epochs):
            eta = scheduler.get_last_lr()[0]
            embeds = model.forward(X, modality, normalize=False)
            cton = 1 - criterion(embeds, Y, dim=1).detach().cpu()
            loss = 1 - criterion(embeds, Y, dim=1)
            update = eta * torch.autograd.grad(outputs=loss.mean(), inputs=X)[0].sign()
            X = (X.detach().cpu() - update.detach().cpu()).to(device)
            X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
            scheduler.step()

        save_image(unnorm(torch.squeeze(X.cuda()))[0], output_path)
        print(f"Saved perturbed image to {output_path}")
