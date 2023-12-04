import torch
import matplotlib.pyplot as plt
from basic_snn import BasicModel
from basic_cnn_copy import BasicModel_CNN
from spikingjelly.activation_based import functional
import numpy as np
import os

ALPHA = 3.7132080089425044
TAU = 2.180830180029865

# def grad_cam(img, label, model):
#     img.requires_grad = True
#     out = model(img)
#     model.zero_grad()
#     out[0, label].backward()
#     gradients = model.get_activations_gradient()
#     pooled_gradients = torch.mean(gradients, dim=[0, 1, 3, 4])
#     activations = model.get_activations(img).detach()
#     for i in range(256):
#         activations[:, :, i, :, :] *= pooled_gradients[i]
#     heatmap = torch.mean(activations, dim=2).squeeze()
#     selected_time_step = -1
#     heatmap = heatmap[selected_time_step]
#     max_value = torch.max(heatmap).item()
#     heatmap = torch.clamp(heatmap, min=0)
#     heatmap /= max_value
#     return heatmap
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

def save_image(image, index, label, pred, folder):
    os.makedirs(folder, exist_ok=True)
    filename = f"{index}_{label}_{pred}.png"
    filepath = os.path.join(folder, filename)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.savefig(filepath)
    plt.close()

def main():
    name = 'deepfool'
    folder_name = f"./result/image/clean/snn/20_{name}/"
    #epsilons=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    epsilons=[0.01, 0.02, 0.05, 0.1]
    #epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]

    datasets = []
    for epsilon in epsilons:
        datasets.append(torch.load(f'./adversarial_example/snn_label/20_{name}_adversarial_{epsilon}.pth'))

    model_snn = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model_snn, step_mode='m')
    model_snn.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))

    # model_cnn = BasicModel_CNN(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    # model_cnn.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))

    model_snn.eval()
    #model_cnn.eval()
    acc = []
    for dataset, epsilon in zip(datasets, epsilons):
        cnt = 0
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs_snn = model_snn(inputs)
                #outputs_cnn = model_cnn(inputs)
                
                _, predicted_snn = torch.max(outputs_snn, 1)
                #_, predicted_cnn = torch.max(outputs_cnn, 1)
                if (labels-predicted_snn).item() != 0:
                    cnt += 1
                for j in range(inputs.size(0)):
                    save_image(inputs[j], i * inputs.size(0) + j, labels[j].item(), predicted_snn[j].item(), folder=os.path.join(folder_name, f'{epsilon}'))
        acc.append(100-cnt)
    print(acc)

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()