import torch
import matplotlib.pyplot as plt
from basic_snn import BasicModel
from basic_cnn_copy import BasicModel_CNN
from spikingjelly.activation_based import functional
import torch.nn.functional as F
import numpy as np
import os

ALPHA = 3.7132080089425044
TAU = 2.180830180029865
#TAU = 3.0

def main():
    name = 'deepfool'
    #epsilons=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    #epsilons = [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]
    iters = [1, 5, 10, 15, 20, 100]
    epsilons = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    model_snn = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model_snn, step_mode='m')
    model_snn.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    model_snn.eval()

    model_cnn = BasicModel_CNN(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model_cnn.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
    model_cnn.eval()

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for iter in iters:
        for epsilon in epsilons:
            datasets = []
            datasets.append(torch.load(f'./adversarial_example/cnn_label/{iter}_{name}_adversarial_cnn_{epsilon}.pth'))
            datasetssnn = []
            datasetssnn.append(torch.load(f'./adversarial_example/snn_label/{iter}_{name}_adversarial_{epsilon}.pth'))

            correct_pred_snn = {classname: 0 for classname in classes}
            total_pred_snn = {classname: 0 for classname in classes}

            correct_pred_cnn = {classname: 0 for classname in classes}
            total_pred_cnn = {classname: 0 for classname in classes}

            for dataset, datasetsnn in zip(datasets, datasetssnn):
                testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
                testloader2 = torch.utils.data.DataLoader(datasetsnn, batch_size=64, shuffle=False, num_workers=0)
                total = 0
                correct = 0
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(testloader):
                        functional.reset_net(model_snn)
                        inputs, labels = inputs.cuda(), labels.cuda()
                        outputs_snn = model_snn(inputs)
                        _, predicted_snn = torch.max(outputs_snn, 1)
                        total += labels.size(0)
                        correct += (predicted_snn == labels).sum().item()
                        for label, prediction in zip(labels, predicted_snn):
                            if label == prediction:
                                correct_pred_snn[classes[label]] += 1
                            total_pred_snn[classes[label]] += 1
                print(f'{iter}_{epsilon} SNN Accuracy of the network on the test images: {100 * correct // total} %')

                # for classname, correct_count in correct_pred_snn.items():
                #     accuracy = 100 * float(correct_count) / total_pred_snn[classname]
                #     print(f'SNN Accuracy for class: {classname:5s} is {accuracy:.1f} %')
                
                total = 0
                correct = 0
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(testloader2):
                        inputs, labels = inputs.cuda(), labels.cuda()
                        outputs_cnn = model_cnn(inputs)
                        _, predicted_cnn = torch.max(outputs_cnn, 1)
                        total += labels.size(0)
                        correct += (predicted_cnn == labels).sum().item()
                        for label, prediction in zip(labels, predicted_cnn):
                            if label == prediction:
                                correct_pred_cnn[classes[label]] += 1
                            total_pred_cnn[classes[label]] += 1
                print(f'{iter}_{epsilon} CNN Accuracy of the network on the test images: {100 * correct // total} %')

                # for classname, correct_count in correct_pred_cnn.items():
                #     accuracy = 100 * float(correct_count) / total_pred_cnn[classname]
                #     print(f'CNN Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()