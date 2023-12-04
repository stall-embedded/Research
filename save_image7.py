import torch
import matplotlib.pyplot as plt
from basic_snn import BasicModel
from basic_cnn_copy import BasicModel_CNN
import torch.nn.functional as F
import numpy as np
import os

def grad_cam(img, label, model):
    model.eval()
    img = img.clone().detach()  # Clone and detach the image tensor
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

def save_image_with_heatmap(input_img, heatmap, index, label, pred, folder, alpha=0.5):
    os.makedirs(folder, exist_ok=True)

    img = input_img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap_resized = F.interpolate(heatmap.unsqueeze(0).unsqueeze(1), size=(32, 32), mode='bicubic', align_corners=False).squeeze().cpu().numpy()
    heatmap_min, heatmap_max = heatmap_resized.min(), heatmap_resized.max()
    if heatmap_max - heatmap_min != 0:
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    else:
        # 분모가 0인 경우에 대한 처리
        heatmap_resized = np.zeros_like(heatmap_resized)

    # 이미지 저장
    plt.imshow(img)
    plt.imshow(heatmap_resized, cmap='hot', alpha=alpha)
    plt.axis('off')  # 축 제거
    filename = f"{index}_{label}_{pred}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    name = 'original'
    model_cnn = BasicModel_CNN(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model_cnn.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
    model_cnn.eval()
    folder_name = f"./result/image/grad_cam/cnn/{name}/"
    datasets = []
    datasets.append(torch.load('original_mini.pth'))

    for dataset in datasets:
        cnt = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs_cnn = model_cnn(inputs)
            
            _, predicted_cnn = torch.max(outputs_cnn, 1)
            if (labels-predicted_cnn).item() != 0:
                cnt += 1
            
            g_cam = grad_cam(inputs, labels, model_cnn)

            for j in range(inputs.size(0)):
                save_image_with_heatmap(inputs[j], g_cam, i, labels[j].item(), predicted_cnn[j].item(), folder_name)
                
        print(100-cnt)

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()