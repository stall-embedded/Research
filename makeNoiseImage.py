import numpy as np
from PIL import Image
import os
from torchvision.datasets import CIFAR10

# Load the CIFAR-10 dataset
dataset = CIFAR10(root='./data', download=True, transform=None)

# Create a directory to save the noisy images
if not os.path.exists('noisy_images'):
    os.makedirs('noisy_images')

# Add noise to the images and save them
for i, (image, label) in enumerate(dataset):
    image = np.array(image)
    noise = np.random.normal(0, 0.1, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)
    noisy_image.save(f'noisy_images/noisy_image_{i}_label_{label}.png')