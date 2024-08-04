import os
import numpy as np

import torch

from utils import unnorm, criterion
from dataset_utils import create_dataset
from models import load_model
import imagebind.data as data
from torchvision.utils import save_image


device='cuda:0'
dataset_flag='imagenet'
modality='vision'
seed=0

def jpeg_transform(X):
    batch_size = X.shape[0]
    transformed_images = []
    for i in range(batch_size):
        image_filename = f'dummy_{i}.jpg'
        save_image(torch.squeeze(unnorm(X[i])), image_filename)
        jpeg_X = data.load_and_transform_vision_data([image_filename], device)
        os.remove(image_filename)
        transformed_images.append(jpeg_X.squeeze().cpu())
    # Stack the transformed images back into a single tensor
    jpeg_X_batch = torch.stack(transformed_images)
    return jpeg_X_batch


def evaluate_jpeg(model, x_adv_path, y_id_path, modality, image_text_dataset):
    x_adv = torch.from_numpy(np.load(x_adv_path))
    y_id = torch.from_numpy(np.load(y_id_path))
    print('Without JPEG defense:')
    with torch.no_grad():
        embeds = model.forward(x_adv.to(device), modality, normalize=False).detach().cpu()
    classes = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].detach().cpu(), dim=2).argsort(dim=1, descending=True)
    print('     Attack success rate: '+str((classes == y_id[:, None])[:, 0].sum().item())+'%')
    adv_loss = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].detach().cpu(), dim=2)[torch.arange(100), y_id]
    adv_loss_avg = np.mean(adv_loss.numpy())
    adv_loss_std = np.std(adv_loss.numpy())
    print("     Average adversarial alignment:", adv_loss_avg)
    print("     Standard deviation of adversarial alignment:", adv_loss_std)
    # attack success rate with jpeg
    print('With JPEG defense')
    with torch.no_grad():
        embeds = model.forward(jpeg_transform(x_adv).to(device), modality, normalize=False).detach().cpu()
    classes = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].detach().cpu(), dim=2).argsort(dim=1, descending=True)
    print('     Attack success rate: '+str((classes == y_id[:, None])[:, 0].sum().item())+'%')
    adv_loss = criterion(embeds[:, None, :].cpu(), image_text_dataset.labels[None, :, :].detach().cpu(), dim=2)[torch.arange(100), y_id]
    adv_loss_avg = np.mean(adv_loss.numpy())
    adv_loss_std = np.std(adv_loss.numpy())
    print("     Average adversarial alignment:", adv_loss_avg)
    print("     Standard deviation of adversarial alignment:", adv_loss_std)

print('-----------Imagebind-----------')
model_flag = 'imagebind'
embs_input = 'outputs/embeddings/imagenet_imagebind_embeddings.npy'
model = load_model(model_flag, device)
image_text_dataset = create_dataset(dataset_flag, model=model, device=device, seed=seed, embs_input=embs_input)
print('Adversarial illusions:')
x_adv_path ='outputs/imagenet/whitebox/imagebind/x_advs_300.npy'
y_id_path = 'outputs/imagenet/whitebox/imagebind/y_ids.npy'
evaluate_jpeg(model,x_adv_path, y_id_path, modality, image_text_dataset)
print()
print('JPEG-resistant Adversarial illusions:')
x_adv_path ='outputs/imagenet/whitebox/imagebind_jpeg/x_advs_300.npy'
y_id_path = 'outputs/imagenet/whitebox/imagebind_jpeg/y_ids.npy'
evaluate_jpeg(model,x_adv_path, y_id_path,modality, image_text_dataset)

print('-----------Audioclip-----------')
model_flag = 'audioclip'
embs_input = 'outputs/embeddings/imagenet_audioclip_embeddings.npy'
model = load_model(model_flag, device)
image_text_dataset = create_dataset(dataset_flag, model=model, device=device, seed=seed, embs_input=embs_input)
print('Adversarial illusions:')
x_adv_path ='outputs/imagenet/whitebox/audioclip/x_advs_300.npy'
y_id_path = 'outputs/imagenet/whitebox/audioclip/y_ids.npy'
evaluate_jpeg(model,x_adv_path, y_id_path,modality, image_text_dataset)
print()
print('JPEG-resistant Adversarial illusions:')
x_adv_path ='outputs/imagenet/whitebox/audioclip_jpeg/x_advs_300.npy'
y_id_path = 'outputs/imagenet/whitebox/audioclip_jpeg/y_ids.npy'
evaluate_jpeg(model,x_adv_path, y_id_path,modality, image_text_dataset)