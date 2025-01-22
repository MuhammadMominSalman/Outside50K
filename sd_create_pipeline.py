import os
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from torchvision import transforms

# Set up paths
OUTPUT_DIR = "Outside50K"
LABELS_1 = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
LABELS_2 = [ "autumn", "winter", "spring", "summer"]
NUM_IMAGES_PER_LABEL = 100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Stable Diffusion Model
print("Loading Stable Diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define prompts for each label
PROMPTS_2 = {
    "sunny": "A beautiful sunny day with clear skies in a natural outdoor setting",
    "cloudy": "A cloudy day with overcast skies in a scenic natural environment",
    "rainy": "A rainy day with visible raindrops and wet surroundings in nature",
    "snowy": "A snowy landscape with snow-covered trees and ground",
}
PROMPTS_1 = {
    "autumn": "An autumn day with colorful fall foliage in an outdoor setting",
    "winter": "A winter day with bare trees and a cold atmosphere",
    "spring": "A spring day with blooming flowers and lush greenery",
    "summer": "A summer day with vibrant sunlight and green landscapes",
}

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((512, 512))
])

# Generate synthetic images
print("Generating images...")
for label_1 in tqdm(LABELS_1):
    for label_2 in LABELS_2:
        prompt = PROMPTS_1[label_1] + PROMPTS_2[label_2]
        label_dir = Path(OUTPUT_DIR)

        for i in range(NUM_IMAGES_PER_LABEL):
            try:
                # Generate image
                image = pipe(prompt).images[0]
                image = transform(image)
                
                # Save image
                image_path = label_dir / f"{label}_{i+1}.png"
                image.save(image_path)
            except Exception as e:
                print(f"Error generating image for {label} #{i+1}: {e}")

print(f"Dataset generation completed. Images saved to {OUTPUT_DIR}")
