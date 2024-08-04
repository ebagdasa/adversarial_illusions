import os
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from utils import threshold, criterion
from models import load_model
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image


# Configure Script
gpu_num = 0
epochs = [100, 200, 300, 500, 1000]
batch_size = 4
eps = 16 / 255
lr = 0.02
n_images = 100
model_flag = 'imagebind'
gamma_epochs = 100
modality = 'thermal'

if type(epochs) == list:
    max_epochs = max(epochs)
else:
    max_epochs = epochs
    epochs = [epochs]

device = f"cuda:{gpu_num}" if torch.cuda.is_available() and gpu_num >= 0 else "cpu"
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(model_flag, device)


def load_and_transform_thermal_data(thermal_paths, device):
    if thermal_paths is None:
        return None

    thermal_ouputs = []
    for thermal_path in thermal_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        with open(thermal_path, "rb") as fopen:
            thermal = Image.open(fopen).convert("L")
        thermal = data_transform(thermal).to(device)
        thermal_ouputs.append(thermal)
    return torch.stack(thermal_ouputs, dim=0)

# Function to classify images
def classify_images(image_dir, target_text, correct_indices):
    Y = model.forward(target_text, "text", normalize=False)
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:50]
    correct_classifications = 0
    total_images = len(image_files)
    
    sim_list = []
    
    with torch.no_grad():
        for image_file in image_files:
            X = load_and_transform_thermal_data([image_file], device)
            embeds = model.forward(X, modality, normalize=True)
            sim = criterion(embeds, Y)            
            index = torch.argmax(sim).item()
            if index in correct_indices:
                correct_classifications += 1
            max_loss = torch.max(sim[correct_indices]).item()
            sim_list.append(max_loss)

    return correct_classifications, total_images, sim_list

def perturb_images(image_dir, target_text, output_subdir, eps=16/255):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:50]
    
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing Images", ncols=100, ascii=True)):
        X = load_and_transform_thermal_data([image_file], device)
        X_init = X.clone().detach().cpu().requires_grad_(False)
        
        Y = model.forward(target_text, "text", normalize=False)
        X, Y = X.to(device).requires_grad_(True), Y.to(device)
        X_max, X_min = threshold(X, eps, modality, device)
        
        optimizer = optim.SGD([X], lr=lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, np.arange(gamma_epochs, max_epochs, gamma_epochs), gamma=0.9)
        
        for j in range(max_epochs):
            eta = scheduler.get_last_lr()[0]
            embeds = model.forward(X, modality, normalize=False)
            cton = 1 - criterion(embeds, Y, dim=1).detach().cpu()
            loss = 1 - criterion(embeds, Y, dim=1)
            update = eta * torch.autograd.grad(outputs=loss.mean(), inputs=X, retain_graph=True)[0].sign()
            X = (X.detach().cpu() - update.detach().cpu()).to(device).requires_grad_(True)
            X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
            scheduler.step()
        # Save the perturbed image
        save_image(X.detach().cpu(), os.path.join(output_subdir, f'perturbed_{idx}.png'))


# Define the image directories
people_images_dir = 'data/thermal_dataset/people'
non_people_images_dir = 'data/thermal_dataset/non-people'

# Target text embeddings (Dummy implementation for demonstration)
target_text = ["person", "man", "woman", "people", "street", "road", "car", "light"]

# Classify people images
people_correct, people_total, people_sim_list = classify_images(people_images_dir, target_text, correct_indices=range(4))
# Classify non-people images
non_people_correct, non_people_total, non_people_sim_list = classify_images(non_people_images_dir, target_text, correct_indices=range(4,8))

print('--------------Organic alignment----------------')
# Print results
total_correct = people_correct + non_people_correct
total_images = people_total + non_people_total
accuracy = (total_correct / total_images) * 100
print(f"Total Correct: {total_correct}")
print(f"Total Images: {total_images}")
print(f"Accuracy: {accuracy:.2f}%")
#concatenate people_sim_list and non_people_sim_list and calculate the mean and std
sim_list = people_sim_list + non_people_sim_list
mean_sim = np.mean(sim_list)
std_sim = np.std(sim_list)
print(f"Mean Similarity: {mean_sim:.4f}")
print(f"Standard Deviation Similarity: {std_sim:.4f}")

print('--------------Adversarial alignment----------------')
print('--------------eps:0----------------')

# Classify people images
people_correct, people_total, people_sim_list = classify_images(people_images_dir, target_text, correct_indices=range(4,8))
# Classify non-people images
non_people_correct, non_people_total, non_people_sim_list = classify_images(non_people_images_dir, target_text, correct_indices=range(4))
# Print results
total_correct = people_correct + non_people_correct
total_images = people_total + non_people_total
accuracy = (total_correct / total_images) * 100
print(f"Total Correct: {total_correct}")
print(f"Total Images: {total_images}")
print(f"Accuracy: {accuracy:.2f}%")
#concatenate people_sim_list and non_people_sim_list and calculate the mean and std
sim_list = people_sim_list + non_people_sim_list
mean_sim = np.mean(sim_list)
std_sim = np.std(sim_list)
print(f"Mean Similarity: {mean_sim:.4f}")
print(f"Standard Deviation Similarity: {std_sim:.4f}")


for eps in [1,4,8,16,32]:
    output_dir = f'outputs/thermal/perturbed_images_eps_{eps}'
    os.makedirs(output_dir, exist_ok=True)
    people_output_dir = os.path.join(output_dir, 'people')
    non_people_output_dir = os.path.join(output_dir, 'non_people')
    os.makedirs(people_output_dir, exist_ok=True)
    os.makedirs(non_people_output_dir, exist_ok=True)
    # Perturb people images to align with 'street'
    perturb_images(people_images_dir, target_text=["street"], output_subdir=people_output_dir, eps=eps/255)
    # Perturb non-people images to align with 'people'
    perturb_images(non_people_images_dir, target_text=["people"], output_subdir=non_people_output_dir, eps=eps/255)

    # Classify people images
    people_correct, people_total, people_sim_list = classify_images(people_output_dir, target_text, correct_indices=range(4,8))
    # Classify non-people images
    non_people_correct, non_people_total, non_people_sim_list = classify_images(non_people_output_dir, target_text, correct_indices=range(4))

    # Print results
    total_correct = people_correct + non_people_correct
    total_images = people_total + non_people_total
    accuracy = (total_correct / total_images) * 100
    print(f'--------------eps:{eps}----------------')

    print(f"Total Correct: {total_correct}")
    print(f"Total Images: {total_images}")
    print(f"Accuracy: {accuracy:.2f}%")
    #concatenate people_sim_list and non_people_sim_list and calculate the mean and std
    sim_list = people_sim_list + non_people_sim_list
    mean_sim = np.mean(sim_list)
    std_sim = np.std(sim_list)
    print(f"Mean Similarity: {mean_sim:.4f}")
    print(f"Standard Deviation Similarity: {std_sim:.4f}")