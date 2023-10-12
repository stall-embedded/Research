import torch
import torchvision
import torchvision.transforms as transforms
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack, LinfDeepFoolAttack, BoundaryAttack, LinfBasicIterativeAttack
import matplotlib.pyplot as plt
import numpy as np
from basic_cnn import BasicModel
import cv2
ALPHA = 0.4
NUM = 1
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
    heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap /= max_value
    return heatmap

def apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name):
    attack = attack_class()
    images = torch.clamp(images, 0, 1)
    _, adversarials, _ = attack(fmodel, images, labels, epsilons=[0.1])

    # adversarials이 텐서인지 확인
    if isinstance(adversarials, list):
        adversarials = adversarials[0]

    print(f'Accuracy on adversarial examples ({attack_name}):', accuracy(fmodel, adversarials, labels))
    heatmap = grad_cam(adversarials.clone().detach().requires_grad_(True), labels[0], model)
    heatmap_resized = cv2.resize(heatmap[NUM], (32, 32))
    
    # 이미지 값이 유효한 범위 내에 있는지 확인
    img_to_display = images[NUM].permute(1, 2, 0).cpu().numpy()
    img_to_display = np.clip(img_to_display, 0, 1)

    plt.imshow(img_to_display)
    plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA)
    plt.title(attack_name)
    plt.show()


def main():
    # Load CIFAR10 dataset
    #testset = torch.load('cifar10_test_dataset.pth')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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
    heatmap_resized = cv2.resize(heatmap[NUM], (32, 32))
    plt.imshow(images[NUM].permute(1, 2, 0).cpu().numpy())
    plt.imshow(heatmap_resized, cmap='hot', alpha=ALPHA)
    plt.title('Original Image')
    plt.show()

    attack_classes = [
        (FGSM, 'FGSM'),
        (PGD, 'PGD'),
        (L2CarliniWagnerAttack, 'C&W'),
        (LinfDeepFoolAttack, 'DeepFool'),
        (BoundaryAttack, 'BoundaryAttack')
    ]

    for attack_class, attack_name in attack_classes:
        apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name)

if __name__ == '__main__':
    main()