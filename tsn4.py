import torch
from basic_snn import BasicModel
import pandas as pd
import matplotlib.pyplot as plt


def main():
    torch.cuda.empty_cache()
    num = 80
    adv_data = torch.load(f'./adversarial_example/cnn_label/deepfool_adversarial_cnn_0.01.pth')
    data = torch.load(f'original_mini.pth')
    first_image = adv_data[num]
    first_image = first_image[0]
    first_image = first_image.permute(1, 2, 0)
    first_image = first_image.detach().cpu()

    second_image = data[num]
    second_image = second_image[0]
    second_image = second_image.permute(1, 2, 0)
    second_image = second_image.detach().cpu()
    fig1 = plt.figure()
    plt.imshow(first_image)

    fig2 = plt.figure()
    plt.imshow(second_image)
    
    plt.show()
if __name__ == '__main__':
    main()
