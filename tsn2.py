import torch
from basic_snn import BasicModel
import pandas as pd
from spikingjelly.activation_based import functional
from collections import Counter

    


def main():
    torch.cuda.empty_cache()
    testsets = []
    epsilons = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    # for epsilon in epsilons:
    #     testsets.append(torch.load(f'./adversarial_example/snn/pgd_adversarial_{epsilon}.pth'))
    dataset = torch.load(f'./adversarial_example/cnn/deepfool_adversarial_cnn_0.01.pth')

    #labels = [label for _, label in dataset]
    #label_counts = Counter(labels)

    #print("라벨 분포:", label_counts)

    print("데이터셋의 샘플 개수:", len(dataset))
    for i in range(min(5, len(dataset))):  # 첫 5개 샘플을 출력
        print(f"샘플 {i}: {dataset[i]}")
        print(dataset.shape)
        print(len(dataset[i][0]))
        print("데이터 타입:", type(dataset[i]))
        if isinstance(dataset[i], tuple):
            print("입력 데이터 타입:", type(dataset[i][0]))
            print("라벨 데이터 타입:", type(dataset[i][1]))
if __name__ == '__main__':
    main()
