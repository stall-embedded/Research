import torch
from basic_snn import BasicModel
import pandas as pd


def main():
    torch.cuda.empty_cache()
    
    #epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    #epsilons = [0.01, 0.02, 0.05, 0.1]
    iters = [1, 5, 10, 15, 20, 100]
    epsilons = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    #c_values = [1e-4, 1e-3, 1e-2, 1e-1]
    #c_values = [5e-1]
    originalset = torch.load('original_mini.pth')
    original_labels = [label for _, label in originalset]
    for iter in iters:
        testsets = []
        snn_testsets = []
        for epsilon in epsilons:
            testsets.append(torch.load(f"./adversarial_example/cnn/{iter}_deepfool_adversarial_cnn_{epsilon}.pth"))
            snn_testsets.append(torch.load(f"./adversarial_example/snn/{iter}_deepfool_adversarial_{epsilon}.pth"))

        for epsilon, testset, snn_testset in zip(epsilons, testsets, snn_testsets):
            labeled_testset = [(data, original_labels[i]) for i, data in enumerate(testset)]
            torch.save(labeled_testset, f"./adversarial_example/cnn_label/{iter}_deepfool_adversarial_cnn_{epsilon}.pth")
            labeled_testset_snn = [(data, original_labels[i]) for i, data in enumerate(snn_testset)]
            torch.save(labeled_testset_snn, f"./adversarial_example/snn_label/{iter}_deepfool_adversarial_{epsilon}.pth")
    
    # testloader = torch.utils.data.DataLoader(originalset, batch_size=8, shuffle=False, num_workers=0)
    # class_images = {i: [] for i in range(10)}
    # for _, (inputs, labels) in enumerate(testloader):
    #     for img, label in zip(inputs, labels):
    #         if len(class_images[label.item()]) < 10:
    #             class_images[label.item()].append(img)

    # image_list = []
    # label_list = []
    # for key, values in class_images.items():
    #     for value in values:
    #         image_list.append(value)  # 이미지만 추가
    #         label_list.append(key)    # 레이블만 추가
    # concatenated_images = torch.stack(image_list, dim=0)
    # print(concatenated_images.shape)
    # result_list = [(img, label) for img, label in zip(concatenated_images, label_list)]
    # torch.save(result_list, f"original_mini.pth")
if __name__ == '__main__':
    main()
