import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack, LinfDeepFoolAttack
import matplotlib.pyplot as plt
import numpy as np
from basic_cnn import BasicModel
import torch.nn.functional as F
import cv2
import os
import pandas as pd

ALPHA = 0.6
NUM = 0
# Grad-CAM function
def grad_cam(img, label, model):
    model.eval()
    img.requires_grad = True
    out = model(img)
    model.zero_grad()
    out[0][label].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(img).detach()
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    max_value = torch.max(heatmap).item()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap /= max_value
    return heatmap

def compute_difference(original_heatmap, attacked_heatmap):
    euclidean_distance = torch.norm(original_heatmap - attacked_heatmap).item()
    cosine_similarity = F.cosine_similarity(original_heatmap.view(1, -1), attacked_heatmap.view(1, -1)).item()
    
    return euclidean_distance, cosine_similarity

def save_image_with_heatmap(image, heatmap, filename):
    img_to_display = image.permute(1, 2, 0).cpu().numpy()
    img_to_display = np.clip(img_to_display, 0, 1)
    heatmap_resized = F.interpolate(heatmap.unsqueeze(0).unsqueeze(1), size=(32, 32), mode='bilinear').squeeze().cpu().numpy()
    
    plt.imshow(img_to_display)
    plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # Load CIFAR10 dataset
    testset = torch.load('cifar10_test_dataset.pth')
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)

    # Assuming model is already trained and loaded as 'model'
    model = BasicModel(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
    model.eval()

    log_df = pd.DataFrame(columns=['Class', 'Image', 'Epsilon', 'Attack', 'Euclidean Distance', 'Cosine Similarity', 'Accuracy'])

    def save_gradient(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    h = model.features[8].register_full_backward_hook(save_gradient)

    # Convert the model to Foolbox model
    fmodel = PyTorchModel(model, bounds=(0, 1))

    # Get some test data
    images, labels = next(iter(testloader))
    images = images.cuda()
    labels = labels.cuda()
    original_accuracy = accuracy(fmodel, images, labels)
    print(f'Accuracy on original images: {original_accuracy}')

    attack_classes = [
        (FGSM, 'FGSM'),
        (PGD, 'PGD'),
        (L2CarliniWagnerAttack, 'C&W'),
        (LinfDeepFoolAttack, 'DeepFool')
    ]

    class_images = {}
    for img, label in zip(images, labels):
        if label.item() not in class_images:
            class_images[label.item()] = []
        class_images[label.item()].append(img)

    os.makedirs("saved_images", exist_ok=True)
    epsilons=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    for label, imgs in class_images.items():
        print(f"Class:{label}")
        imgs_tensor = torch.stack(imgs).cuda()
        imgs_tensor = torch.clamp(imgs_tensor, 0, 1)
        acc = accuracy(fmodel, imgs_tensor, torch.tensor([label], device='cuda').repeat(len(imgs)))
        temp_df = pd.DataFrame({'Class': [label],
                                'Image': ['N/A'],
                                'Epsilon': ['N/A'], 
                                'Attack': ['Original'], 
                                'Euclidean Distance': ['N/A'], 
                                'Cosine Similarity': ['N/A'], 
                                'Accuracy': [acc]})
        log_df = pd.concat([log_df, temp_df], ignore_index=True)
        for attack_class, attack_name in attack_classes:
            print(f"attack:{attack_name}")
            for epsilon in epsilons:
                print(f"epsilon:{epsilon}")
                _, adversarials, _ = attack_class()(fmodel, imgs_tensor, torch.tensor([label], device='cuda').repeat(len(imgs_tensor)), epsilons=epsilon)
                if isinstance(adversarials, list):
                    adversarials = adversarials[0]
                acc_adv = accuracy(fmodel, adversarials, labels[:len(adversarials)])
                for idx, img in enumerate(imgs):
                    if epsilon == epsilons[0]:
                        img = img.unsqueeze(0).cuda()
                        img = torch.clamp(img, 0, 1)
                        heatmap = grad_cam(img.clone().detach().requires_grad_(True), label, model)
                        save_image_with_heatmap(img[0], heatmap, f"saved_images/original_class_{label}_img_{idx}.png")
                for idx, adv_img in enumerate(adversarials):
                    if idx < 5:
                        heatmap_adversarial = grad_cam(adv_img.unsqueeze(0).clone().detach().requires_grad_(True), label, model)
                        save_image_with_heatmap(imgs[idx], heatmap_adversarial, f"saved_images/{attack_name}_class_{label}_img_{idx}_eps_{epsilon}.png")
                        save_image_with_heatmap(adv_img, heatmap_adversarial, f"saved_images/adv_{attack_name}_class_{label}_img_{idx}_eps_{epsilon}.png")
                        euclidean_distance, cosine_similarity = compute_difference(heatmap, heatmap_adversarial)
                        print(f"Class {label}, Image {idx}, EPS {epsilon}, Attack {attack_name}: Euclidean Distance = {euclidean_distance}, Cosine Similarity = {cosine_similarity}")

                        temp_df = pd.DataFrame({'Class': [label],
                                                'Image': [idx], 
                                                'Epsilon': [epsilon], 
                                                'Attack': [attack_name], 
                                                'Euclidean Distance': [euclidean_distance], 
                                                'Cosine Similarity': [cosine_similarity], 
                                                'Accuracy': [acc_adv]})
                        log_df = pd.concat([log_df, temp_df], ignore_index=True)
    
    log_df.to_csv("attack_log_cnn.csv", index=False)

if __name__ == '__main__':
    main()