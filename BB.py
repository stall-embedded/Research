import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from basic_snn import BasicModel
import torch.nn.functional as F
import cv2
import os
import pandas as pd
from spikingjelly.activation_based import functional, neuron

ALPHA_pic = 0.6
NUM = 0
ALPHA = 3.7132080089425044
TAU = 2.180830180029865

def grad_cam(img, label, model):
    img.requires_grad = True
    out = model(img)
    model.zero_grad()
    out[0, label].backward()
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
    originalset = torch.load('adversarial_example/snn/original.pth')
    dataset = torch.load('adversarial_example/snn/cw_adversarial.pth')

    model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model, step_mode='m')
    model.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    model.train()
    
    results = []
    for data in originalset:  # dataset은 여러분의 데이터셋을 나타냅니다.
        img, label = data  # 이미지와 레이블을 얻습니다.
        heatmap = grad_cam(img, label, model)  # Grad-CAM 적용

        prediction = model(img).argmax().item()
        attacked_heatmap = grad_cam(attacked_img, attacked_label, model)

        euclidean_distance, cosine_similarity = compute_difference(heatmap, attacked_heatmap)

        results.append([label, prediction, euclidean_distance, cosine_similarity])

    # 히트맵과 원본 이미지를 겹쳐서 저장
        save_image_with_heatmap(img, heatmap, f"heatmap_{label}.png")

    # 결과를 CSV 파일로 저장
    df = pd.DataFrame(results, columns=['Label', 'Prediction', 'Euclidean Distance', 'Cosine Similarity'])
    df.to_csv('results.csv', index=False)

    def save_gradient(module, grad_input, grad_output):
        model.gradients = grad_output[0]

    h = model.features[8].register_full_backward_hook(save_gradient)


if __name__ == '__main__':
    main()