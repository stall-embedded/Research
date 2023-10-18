import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack, LinfDeepFoolAttack
import matplotlib.pyplot as plt
import numpy as np
from basic_snn import BasicModel
import torch.nn.functional as F
import cv2
from spikingjelly.activation_based import functional, neuron
ALPHA_pic = 0.6
NUM = 0
ALPHA = 3.7132080089425044
TAU = 2.180830180029865

def grad_cam(img, label, model):
    img.requires_grad = True
    out = model(img)
    model.zero_grad()
    out[0, label].backward(retain_graph=True)
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 1, 3, 4])
    activations = model.get_activations(img).detach()
    for i in range(256):
        activations[:, :, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=2).squeeze()
    selected_time_step = -1
    heatmap = heatmap[selected_time_step]
    max_value = torch.max(heatmap).item()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap /= max_value
    functional.reset_net(model)
    return heatmap

def compute_difference(original_heatmap, attacked_heatmap):
    euclidean_distance = torch.norm(original_heatmap - attacked_heatmap).item()
    cosine_similarity = F.cosine_similarity(original_heatmap.view(1, -1), attacked_heatmap.view(1, -1)).item()
    return euclidean_distance, cosine_similarity

def apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name):
    torch.cuda.empty_cache()
    attack = attack_class()
    images = torch.clamp(images, 0, 1)
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    original_heatmap = grad_cam(images.clone().detach().requires_grad_(True), labels[0], model)
    for epsilon in epsilons:
        _, adversarials, _ = attack(fmodel, images, labels, epsilons=[epsilon])
        if isinstance(adversarials, list):
            adversarials = adversarials[0]
        print(f'Accuracy on adversarial examples ({attack_name}, epsilon={epsilon}):', accuracy(fmodel, adversarials, labels))
        attacked_heatmap = grad_cam(adversarials.clone().detach().requires_grad_(True), labels[0], model)
        heatmap_resized = F.interpolate(attacked_heatmap[NUM].unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear').squeeze().cpu().numpy()
        img_to_display = images[NUM].permute(1, 2, 0).cpu().numpy()
        img_to_display = np.clip(img_to_display, 0, 1)
        adv_img_to_display = adversarials[NUM].permute(1, 2, 0).cpu().numpy()
        adv_img_to_display = np.clip(adv_img_to_display, 0, 1)
        plt.imshow(img_to_display)
        plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA_pic)
        plt.title(f'{attack_name} (epsilon={epsilon})')
        plt.show()
        euclidean_dist, cosine_sim = compute_difference(original_heatmap, attacked_heatmap)
        print(f"Euclidean Distance(epsilon={epsilon}) : {euclidean_dist}\nCosine Similarity between original and attacked heatmap (epsilon={epsilon}): {cosine_sim}")
        functional.reset_net(model)
        fmodel = PyTorchModel(model, bounds=(0, 1))


def main():
    testset = torch.load('cifar10_test_dataset.pth')
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=6)

    model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model, step_mode='m')
    model.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    model.train()

    def save_gradient(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    h = model.features[8].register_forward_hook(model.forward_hook)
    h_back = model.features[8].register_full_backward_hook(model.backward_hook)

    fmodel = PyTorchModel(model, bounds=(0, 1))

    images, labels = next(iter(testloader))
    images = images.cuda()
    labels = labels.cuda()
    heatmap = grad_cam(images.clone().detach().requires_grad_(True), labels[0], model)
    heatmap_resized = F.interpolate(heatmap[NUM].unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear').squeeze().cpu().numpy()
    plt.imshow(images[NUM].permute(1, 2, 0).cpu().numpy())
    plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA_pic)
    plt.title('Original Image')
    plt.show()
    original_accuracy = accuracy(fmodel, images, labels)
    print(f'Accuracy on original images: {original_accuracy}')

    attack_classes = [
        (FGSM, 'FGSM'),
        (PGD, 'PGD'),
        (L2CarliniWagnerAttack, 'C&W'),
        (LinfDeepFoolAttack, 'DeepFool')
    ]

    for attack_class, attack_name in attack_classes:
        apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name)

    h.remove()
    h_back.remove()

if __name__ == '__main__':
    main()