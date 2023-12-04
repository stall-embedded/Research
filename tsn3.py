import torch
from basic_snn import BasicModel
import pandas as pd


def main():
    torch.cuda.empty_cache()
    originalset = torch.load('original_mini.pth')
    original_labels = [label for _, label in originalset]
    
    testset = torch.load(f'./adversarial_example/cnn/cw_adversarial_cnn.pth')
    #testset = torch.reshape(testset, (100, 3, 32, 32))
    print("Length of originalset:", len(originalset))
    print("Length of testset:", len(testset))
    labeled_testset = [(data, original_labels[i]) for i, data in enumerate(testset)]
    torch.save(labeled_testset, f"./adversarial_example/cnn_label/cw_adversarial_cnn.pth")
if __name__ == '__main__':
    main()
