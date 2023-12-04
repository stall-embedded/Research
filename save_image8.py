import torch
import matplotlib.pyplot as plt
from basic_snn import BasicModel
from spikingjelly.activation_based import functional
import torch.nn.functional as F
import numpy as np
import os

ALPHA = 3.7132080089425044
TAU = 2.180830180029865
def calculate_ncs(activations, decay_factor=0.5):
    time_steps = activations.shape[0]
    ncs = torch.zeros_like(activations)
    last_spike_time = torch.full_like(activations[0], -float('inf'))  # 마지막 스파이크 시간 초기화

    for t in range(time_steps):
        spike = activations[t] > 0  # 스파이크 발생 여부 확인
        last_spike_time[spike] = t  # 스파이크가 발생한 뉴런의 마지막 스파이크 시간 업데이트

        time_since_last_spike = t - last_spike_time  # 마지막 스파이크 이후의 시간 계산
        decay = torch.exp(-decay_factor * time_since_last_spike)
        ncs[t] = decay * activations[t]

    return ncs

def compute_sam(ncs):
    sam = torch.sum(ncs, dim=1)  # Sum over channel dimension
    sam = torch.clamp(sam, min=0)  # Remove negative values
    max_value = torch.max(sam).item()
    if max_value > 0:
        sam /= max_value  # Normalize
    return sam

def compute_sam_per_time_step(ncs):
    sam_per_time_step = []
    selected_time_steps = []
    for t in range(ncs.shape[0]):
        if t%7 == 0:
            sam = torch.sum(ncs[t], dim=1)  # 채널 차원을 따라 합산
            sam = torch.clamp(sam, min=0)
            max_value = torch.max(sam).item()
            if max_value > 0:
                sam /= max_value
            sam_per_time_step.append(sam)
            selected_time_steps.append(t)
    return sam_per_time_step, selected_time_steps

def save_image_with_heatmap(input_img, heatmaps, time_steps, index, label, pred, folder, alpha=0.5):
    os.makedirs(folder, exist_ok=True)
    base_folder = os.path.join(folder, f"{index}_{label}_{pred}")
    os.makedirs(base_folder, exist_ok=True)

    # 이미지 정규화 및 변환
    img = input_img.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    for t, heatmap in zip(time_steps, heatmaps):
        # 히트맵 정규화 및 컬러맵 적용
        #heatmap = heatmap.cpu().numpy()
        #heatmap = torch.from_numpy(heatmap) if isinstance(heatmap, np.ndarray) else heatmap
        #heatmap_resized = F.interpolate(heatmap.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        heatmap_resized = F.interpolate(heatmap.unsqueeze(0), size=(32, 32), mode='bicubic', align_corners=False).squeeze().cpu().numpy()
        heatmap_min, heatmap_max = heatmap_resized.min(), heatmap_resized.max()
        heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)

        # 이미지 저장
        plt.imshow(img)
        plt.imshow(heatmap_resized, cmap='hot', alpha=alpha)
        plt.axis('off')  # 축 제거
        filename = f"time_step_{t}.png"
        filepath = os.path.join(base_folder, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    name = 'original'
    folder_name = f"./result/image/sam/snn/{name}/"
    datasets = []
    datasets.append(torch.load('original_mini.pth'))

    model_snn = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model_snn, step_mode='m')
    model_snn.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    
    for dataset in datasets:
        cnt = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                functional.reset_net(model_snn)
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs_snn = model_snn(inputs)
                _, predicted_snn = torch.max(outputs_snn, 1)
                if (labels-predicted_snn).item() != 0:
                    cnt += 1

                for j in range(inputs.size(0)):
                    functional.reset_net(model_snn)
                    image_index = i * inputs.size(0) + j
                    activations = model_snn.get_activations(inputs[j]).detach()
                    ncs = calculate_ncs(activations)
                    sam, times = compute_sam_per_time_step(ncs)
                    save_image_with_heatmap(inputs[j], sam, times, image_index, labels[j].item(), predicted_snn[j].item(), folder=folder_name)

        print(100-cnt)

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()