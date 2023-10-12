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

ALPHA = 0.5
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
        if len(class_images[label.item()]) < 3:
            class_images[label.item()].append(img)

    os.makedirs("saved_images", exist_ok=True)

    for label, imgs in class_images.items():
        for idx, img in enumerate(imgs):
            img = img.unsqueeze(0).cuda()
            img = torch.clamp(img, 0, 1)
            heatmap = grad_cam(img.clone().detach().requires_grad_(True), label, model)
            save_image_with_heatmap(img[0], heatmap, f"saved_images/original_class_{label}_img_{idx}.png")
            
            for attack_class, attack_name in attack_classes:
                _, adversarials, _ = attack_class()(fmodel, img, torch.tensor([label]).cuda(), epsilons=[0.01])
                if isinstance(adversarials, list):
                    adversarials = adversarials[0]
                print(f'Accuracy on adversarial examples ({attack_name}):', accuracy(fmodel, adversarials, labels))
                heatmap_adversarial = grad_cam(adversarials.clone().detach().requires_grad_(True), label, model)
                save_image_with_heatmap(adversarials[0], heatmap_adversarial, f"saved_images/{attack_name}_class_{label}_img_{idx}.png")
                euclidean_distance, cosine_similarity = compute_difference(heatmap, heatmap_adversarial)
                print(f"Class {label}, Image {idx}, Attack {attack_name}: Euclidean Distance = {euclidean_distance}, Cosine Similarity = {cosine_similarity}")

if __name__ == '__main__':
    main()