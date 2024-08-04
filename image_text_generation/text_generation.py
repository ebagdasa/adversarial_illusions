from transformers import AutoModel, AutoTokenizer
import torch
from model.openllama import OpenLLAMAPEFTModel
import os
from tqdm import tqdm
import json

# Initialize the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/converted/vicuna_full',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pandagpt_7b_max_len_1024/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
print("Initializing model...")
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print("Model initialized.")

def generate_response(prompt_text, image_path=None, audio_path=None, video_path=None, thermal_path=None, top_p=0.01, temperature=1.0, max_length=128):
    """Generate a response from the model."""
    response = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'audio_paths': [audio_path] if audio_path else [],
        'video_paths': [video_path] if video_path else [],
        'thermal_paths': [thermal_path] if thermal_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': []

    })
    return response

responses = {}
prompt_text = "Can you describe the image?"

for filename in tqdm(os.listdir("../../data/thermal_dataset/raw_images/")):
    if filename.endswith(".jpg"): 
        image_path = os.path.join("../../data/thermal_dataset/raw_images/", filename)
        response = generate_response(prompt_text, image_path=image_path, top_p=0.1, temperature=1.0, max_length=128)
        # print(f"Response for {filename}:", response)
        responses[filename] = response
with open("../../outputs/thermal/perturbed_images_text_generation/response_thermal_unperturbed.json", "w") as f:
    json.dump(responses, f)
print("Responses saved to ../../outputs/thermal/perturbed_images_text_generation/response_thermal_unperturbed.json")

responses = {}
for filename in tqdm(os.listdir("../../outputs/thermal/perturbed_images_text_generation/")):
    if filename.endswith(".png"):  # Assuming all images are PNGs
        image_path = os.path.join("../../outputs/thermal/perturbed_images_text_generation/", filename)
        response = generate_response(prompt_text, image_path=image_path, top_p=0.1, temperature=1.0, max_length=128)
        # print(f"Response for {filename}:", response)
        responses[filename] = response
with open("../../outputs/thermal/perturbed_images_text_generation/response_thermal_perturbed.json", "w") as f:
    json.dump(responses, f)
print("Responses saved to ../../outputs/thermal/perturbed_images_text_generation/response_thermal_perturbed.json")