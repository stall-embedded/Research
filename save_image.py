import torch
import matplotlib.pyplot as plt
from basic_snn import BasicModel
from basic_cnn_copy import BasicModel_CNN
from spikingjelly.activation_based import functional
import os

ALPHA = 3.7132080089425044
TAU = 2.180830180029865

# 이미지 저장 함수

def save_image(image, index, label, pred_snn, pred_cnn, folder="./result/image/original"):
    os.makedirs(folder, exist_ok=True)
    filename = f"{index}_{label}_{pred_snn}_{pred_cnn}.png"
    filepath = os.path.join(folder, filename)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.savefig(filepath)
    plt.close()

def main():
    dataset = torch.load('original_mini.pth')
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_snn = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model_snn, step_mode='m')
    model_snn.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))

    model_cnn = BasicModel_CNN(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model_cnn.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))

    model_snn.eval()
    model_cnn.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs_snn = model_snn(inputs)
            outputs_cnn = model_cnn(inputs)
            
            _, predicted_snn = torch.max(outputs_snn, 1)
            _, predicted_cnn = torch.max(outputs_cnn, 1)

            for j in range(inputs.size(0)):
                save_image(inputs[j], i * inputs.size(0) + j, labels[j].item(), predicted_snn[j].item(), predicted_cnn[j].item())

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()