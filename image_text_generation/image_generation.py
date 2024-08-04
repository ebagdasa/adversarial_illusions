# Standard Libraries
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
import requests

# PyTorch and Related Libraries
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Custom Modules
import image_bind.data as data
from image_bind.models import imagebind_model
from image_bind.models.imagebind_model import ModalityType
from ldm.models.diffusion.ddpm import ImageEmbeddingConditionedLatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler

# Configuration Management
from omegaconf import OmegaConf
from easydict import EasyDict

experiment=input("Enter the experiment name (whitebox, query, query_full): ")
if experiment=='whitebox':
    original_tensor_dir="../data/imagenet/image_generation_data/x_inits.npy"
    original_image_dir="../outputs/imagenet/image_generation/whitebox/original_image/"
    generated_original_dir="../outputs/imagenet/image_generation/whitebox/generated_original_image/"
    perturbed_tensor_dir="../data/imagenet/image_generation_data/x_advs.npy"
    perturbed_image_dir="../outputs/imagenet/image_generation/whitebox/perturbed_images/"
    generated_perturbed_dir="../outputs/imagenet/image_generation/whitebox/generated_perturbed_image/"
elif experiment=='query':
    original_tensor_dir="../data/imagenet/query_image_generation_data/x_inits.npy"
    original_image_dir="../outputs/imagenet/image_generation/query/original_image/"
    generated_original_dir="../outputs/imagenet/image_generation/query/generated_original_image/"
    perturbed_tensor_dir="../data/imagenet/query_image_generation_data/x_advs.npy"
    perturbed_image_dir="../outputs/imagenet/image_generation/query/perturbed_images/"
    generated_perturbed_dir="../outputs/imagenet/image_generation/query/generated_perturbed_image/"
elif experiment=='query_full':
    original_tensor_dir="../data/imagenet/query_full_image_generation_data/x_inits.npy"
    original_image_dir="../outputs/imagenet/image_generation/query_full/original_image/"
    generated_original_dir="../outputs/imagenet/image_generation/query_full/generated_original_image/"
    perturbed_tensor_dir="../data/imagenet/query_full_image_generation_data/x_advs.npy"
    perturbed_image_dir="../outputs/imagenet/image_generation/query_full/perturbed_images/"
    generated_perturbed_dir="../outputs/imagenet/image_generation/query_full/generated_perturbed_image/"


class Binder:
    """ Wrapper for ImageBind model
    """
    def __init__(self, pth_path, device='cuda'):
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.device = device
        self.model.eval()
        self.model.to(device)

        self.data_process_dict = {ModalityType.TEXT: data.load_and_transform_text,
                                  ModalityType.VISION: data.load_and_transform_vision_data,
                                  ModalityType.AUDIO: data.load_and_transform_audio_data}

    def run(self, ctype, cpaths, post_process=False):
        """ ctype: str
            cpaths: list[str]
        """
        inputs = {ctype: self.data_process_dict[ctype](cpaths, self.device)}
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ctype]
        
    def run_tensor(self, ctype, image_tensor):
        """ ctype: str
        """
        inputs = {ctype: image_tensor}
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ctype]
        
device = 'cpu'
binder = Binder(pth_path="../.checkpoints/imagebind_huge.pth", device=device)

# options
opt = EasyDict(config = './configs/stable-diffusion/v2-1-stable-unclip-h-bind-inference.yaml',
               device = 'cuda:0',
               ckpt = './checkpoints/stable-diffusion-2-1-unclip/sd21-unclip-h.ckpt',
               C = 4,
               H = 768,
               W = 768,
               f = 8,
               steps = 50, 
               n_samples = 1,
               scale = 20,
               ddim_eta = 0,
                torch_dtype=torch.float16
               )

config = OmegaConf.load(f"{opt.config}")
shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
batch_size = opt.n_samples

# prepare diffusion model
model = ImageEmbeddingConditionedLatentDiffusion(**config.model['params'])
pl_sd = torch.load(opt.ckpt, map_location="cpu")
sd = pl_sd["state_dict"]
model.load_state_dict(sd, strict=False)
model= model.half()
model.to(opt.device)
model.eval()

sampler = DDIMSampler(model, device=opt.device)

image_model = imagebind_model.imagebind_huge(pretrained=True)
image_model.to('cpu')
image_model.eval()

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

def generate_image_image(embeddings):
    prompts = ['colorful, DSLR quality, clear, vivid'] * batch_size    # you may add extra descriptions you like here
    # c_adm = binder.run(ctype='audio', cpaths=['assets/bird_audio.wav'])
    c_adm = embeddings / embeddings.norm() * 20   # a norm of 20 typically gives better result 
    c_adm = torch.cat([c_adm] * batch_size, dim=0)
    c_adm = c_adm.half().to('cuda:0')
    n_prompt = 'watermark, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    with torch.no_grad(), torch.autocast('cuda'):
        c_adm, noise_level_emb = model.noise_augmentor(c_adm, noise_level=torch.zeros(batch_size).long().to(c_adm.device))
        c_adm = torch.cat((c_adm, noise_level_emb), 1)
    
        uc = model.get_learned_conditioning(batch_size * [n_prompt])    # negative prompts
        uc = {"c_crossattn": [uc], "c_adm": torch.zeros_like(c_adm)}
        c = {"c_crossattn": [model.get_learned_conditioning(prompts)], "c_adm": c_adm}
        
        samples, _ = sampler.sample(S=opt.steps,
                                    conditioning=c,
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=opt.scale,
                                    unconditional_conditioning=uc,
                                    eta=opt.ddim_eta,
                                    x_T=None)
    
    x_samples = model.decode_first_stage(samples.half())
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    # plt.imshow(x_samples[0].permute(1,2,0).cpu().float().numpy())
    return x_samples

def save_perturbed_image(input_dir,output_dir):
    image_tensor=torch.tensor(np.load(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(image_tensor):
        save_image(torch.squeeze(unnorm(img)), os.path.join(output_dir, f'image_{i}.png'))

def save_generated_image(input_dir,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(100):
        image_path = os.path.join(input_dir, f'image_{i}.png')
        inputs = {'vision': data.load_and_transform_vision_data([image_path], 'cpu')}
        embeddings = binder.model(inputs)
        x_samples = generate_image_image(embeddings['vision'])
        save_image(x_samples[0], os.path.join(output_dir, f'image_{i}.png'))



save_perturbed_image(original_tensor_dir, original_image_dir)
save_generated_image(original_image_dir, generated_original_dir)
save_perturbed_image(perturbed_tensor_dir, perturbed_image_dir)
save_generated_image(perturbed_image_dir, generated_perturbed_dir)

# Load the numpy arrays from file
orig_label_path = '../data/imagenet/image_generation_data/y_origs.npy'
y_ids_path = '../data/imagenet/image_generation_data/y_ids.npy'
orig_labels = np.load(orig_label_path)
y_ids = np.load(y_ids_path)
# Convert numpy arrays to lists of integers
orig_labels_list = orig_labels.tolist()
target_labels_list = y_ids.tolist()

# ImageNet class labels
IMAGENET_LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
class_idx = json.loads(requests.get(IMAGENET_LABELS_URL).text)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
def vit_prediction(image_path):
    """Classify image using Vision Transformer and return top-5 labels and indices."""
    image = Image.open(image_path)
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = vit_model(**inputs).logits
        # Get the top-5 predictions
        _, predicted_indices = torch.topk(logits, 5)
        predicted_labels = [idx2label[idx] for idx in predicted_indices[0]]

    return predicted_labels, predicted_indices[0].tolist()

def classify_image(image_directory):
    labels={}
    results=[]
    for i in range(100): 
        image_filename = f"image_{i}.png"
        image_path = os.path.join(image_directory, image_filename)
        if os.path.isfile(image_path):
            label, result = vit_prediction(image_path)
            results.append(label)
            labels[image_filename] = result
        else:
            print(f"The file {image_path} does not exist.")
    return labels, results

def calculate_accuracy(perdicted_labels):
        original_label_top1=0
        original_label_top5=0
        target_label_top1=0
        target_label_top5=0
        for i, (image_name, label_list) in enumerate(perdicted_labels.items()):
                if orig_labels_list[i] == label_list[0]:
                        original_label_top1 += 1     
                if orig_labels_list[i] in label_list:
                        original_label_top5 += 1       
                if target_labels_list[i] == label_list[0]:
                        target_label_top1 += 1
                if target_labels_list[i] in label_list:
                        target_label_top5 += 1

        original_label_top1_accuracy = (original_label_top1 / 100) * 100
        original_label_top5_accuracy = (original_label_top5 / 100) * 100
        target_label_top1_accuracy = (target_label_top1 / 100) * 100
        target_label_top5_accuracy = (target_label_top5 / 100) * 100

        print(f"The top-1 accuracy for the original label is: {original_label_top1_accuracy:.2f}%")
        print(f"The top-5 accuracy for the original label is: {original_label_top5_accuracy:.2f}%")
        print(f"The top-1 accuracy for the target label is: {target_label_top1_accuracy:.2f}%")
        print(f"The top-5 accuracy for the target label is: {target_label_top5_accuracy:.2f}%")

labels_original, results_orginial = classify_image(original_image_dir)
labels_generated_original, results_generated_orginial = classify_image(generated_original_dir)
labels_perturbed, results_perturbed = classify_image(perturbed_image_dir)
labels_generated_perturbed, results_generated_perturbed = classify_image(generated_perturbed_dir)

print("-----------Original image-----------")
calculate_accuracy(labels_original)
print("-----------Generated image from original image-----------")
calculate_accuracy(labels_generated_original)
print("-----------Perturbed image-----------")
calculate_accuracy(labels_perturbed)
print("-----------Generated image from perturbed image-----------")
calculate_accuracy(labels_generated_perturbed)