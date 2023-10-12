import torch
import torchvision
import torchvision.transforms as transforms
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import FGSM, PGD, L2CarliniWagnerAttack, LinfDeepFoolAttack, BoundaryAttack, LinfBasicIterativeAttack
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np
from basic_cnn import BasicModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def apply_attack_and_visualize(attack_class, fmodel, images, labels, model, attack_name):
    attack = attack_class()
    adversarials = attack(fmodel, images, labels, epsilons=[0.1])
    print(f'Accuracy on adversarial examples ({attack_name}):', accuracy(fmodel, adversarials, labels))
    
    target_layers = [model.features[8]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(label.item()) for label in labels]
    grayscale_cam = cam(input_tensor=adversarials, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(adversarials[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam)
    plt.imshow(visualization)
    plt.title(attack_name)
    plt.show()

def main():
    # CIFAR10 데이터셋 로드
    testset = torch.load('cifar10_test_dataset.pth')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)
    # 모델 로드
    model = BasicModel(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
    model.eval()
    # Foolbox 모델로 변환
    fmodel = PyTorchModel(model, bounds=(0, 1))
    # 테스트 데이터 가져오기
    images, labels = next(iter(testloader))
    images = images.cuda()
    labels = labels.cuda()
    images = images.clamp(0, 1).float()

    target_layers = [model.features[8]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(label.item()) for label in labels]
    grayscale_cam_original = cam(input_tensor=images, targets=targets)
    visualization_original = show_cam_on_image(images[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam_original[0, :])
    plt.imshow(visualization_original)
    plt.title("Original Image with CAM")
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