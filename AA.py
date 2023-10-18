import torch
import matplotlib.pyplot as plt
import numpy as np
from basic_snn import BasicModel
import MultiAdversarialAttack as MAA
from spikingjelly.activation_based import functional
import os

ALPHA = 3.7132080089425044
TAU = 2.180830180029865

def main():
    testset = torch.load('cifar10_test_dataset.pth')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)
    model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model, step_mode='m')
    model.load_state_dict(torch.load('best_basic_model_snn_sj0929.pth'))
    model.train()
    images, labels = next(iter(testloader))
    images = images.cuda()
    labels = labels.cuda()
    
    if not os.path.exists('./adversarial_example'):
        os.makedirs('./adversarial_example')

    attacker = MAA.MultiAdversarialAttack(model, 'cuda')
    epsilons=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    for epsilon in epsilons:
        print(f"Generating adversarial examples for epsilon: {epsilon}")
        attacker.generate_and_save("fgsm", images, labels, f"./adversarial_example/fgsm_adversarial_{epsilon}.pth", epsilon=epsilon)
        attacker.generate_and_save("pgd", images, labels, f"./adversarial_example/pgd_adversarial_{epsilon}.pth", epsilon=epsilon, alpha=0.01, iters=100)
        attacker.generate_and_save("boundary", images, labels, f"./adversarial_example/boundary_adversarial_{epsilon}.pth", epsilon=epsilon)

    attacker.generate_and_save("c&w", images, labels, f"./adversarial_example/c&w_adversarial.pth")
    attacker.generate_and_save("deepfool", images, labels, f"./adversarial_example/deepfool_adversarial.pth")
    attacker.generate_and_save("jsma", images, labels, f"./adversarial_example/jsma_adversarial.pth")

if __name__ == '__main__':
    main()