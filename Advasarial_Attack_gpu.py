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
ALPHA = 0.4
NUM = 9
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

def apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name):
    attack = attack_class()
    images = torch.clamp(images, 0, 1)
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    original_heatmap = grad_cam(images.clone().detach().requires_grad_(True), labels[0], model)
    for epsilon in epsilons:
        _, adversarials, _ = attack(fmodel, images, labels, epsilons=[epsilon])

        # adversarials이 텐서인지 확인
        if isinstance(adversarials, list):
            adversarials = adversarials[0]

        print(f'Accuracy on adversarial examples ({attack_name}, epsilon={epsilon}):', accuracy(fmodel, adversarials, labels))
        attacked_heatmap = grad_cam(adversarials.clone().detach().requires_grad_(True), labels[0], model)
        heatmap_resized = F.interpolate(attacked_heatmap[NUM].unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear').squeeze().cpu().numpy()
        
        # 이미지 값이 유효한 범위 내에 있는지 확인
        # img_to_display = images[NUM].permute(1, 2, 0).cpu().numpy()
        # img_to_display = np.clip(img_to_display, 0, 1)
        adv_img_to_display = adversarials[NUM].permute(1, 2, 0).cpu().numpy()
        adv_img_to_display = np.clip(adv_img_to_display, 0, 1)

        plt.imshow(adv_img_to_display)
        plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA)
        plt.title(f'{attack_name} (epsilon={epsilon})')
        plt.show()

        euclidean_dist, cosine_sim = compute_difference(original_heatmap, attacked_heatmap)
        print(f"Euclidean Distance(epsilon={epsilon}) : {euclidean_dist}\nCosine Similarity between original and attacked heatmap (epsilon={epsilon}): {cosine_sim}")


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
    heatmap = grad_cam(images.clone().detach().requires_grad_(True), labels[0], model)
    heatmap_resized = F.interpolate(heatmap[NUM].unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear').squeeze().cpu().numpy()
    plt.imshow(images[NUM].permute(1, 2, 0).cpu().numpy())
    plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA)
    plt.title('Original Image')
    plt.show()
    original_accuracy = accuracy(fmodel, images, labels)
    print(f'Accuracy on original images: {original_accuracy}')

    attack_classes = [
        (FGSM, 'FGSM'),
        (PGD, 'PGD'),
        #(L2CarliniWagnerAttack, 'C&W'),
        (LinfDeepFoolAttack, 'DeepFool')
    ]

    for attack_class, attack_name in attack_classes:
        apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name)

if __name__ == '__main__':
    main()