import torch
import matplotlib.pyplot as plt
import numpy as np
from basic_cnn import BasicModel
import MultiAdversarialAttack_CNN as MAA
import os
from deepfool_copy import DeepFool


def main():
    torch.cuda.empty_cache()
    #testset = torch.load('cifar10_test_dataset.pth')
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)
    model = BasicModel(num_channels=3, optimizer="Adam", lr=0.001).cuda()
    model.load_state_dict(torch.load('best_basic_model_cnn_relu.pth'))
    model.train()
    
    if not os.path.exists('./adversarial_example/cnn'):
        os.makedirs('./adversarial_example/cnn')
    dataset = torch.load('original_mini.pth')
    testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    total_train_batches = len(testloader)
    attacker = MAA.MultiAdversarialAttack(model, 'cuda')
    # testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)
    # class_images = {i: [] for i in range(10)}
    # for _, (inputs, labels) in enumerate(testloader):
    #     for img, label in zip(inputs, labels):
    #         if len(class_images[label.item()]) < 10:
    #             class_images[label.item()].append(img)

    # overshoots = [0.01, 0.02, 0.05, 0.1]
    # for overshoot in overshoots:
    #     deepfool_adversarial_images = []
    #     for i, (images, labels) in enumerate(testloader):
    #         print(f"{i}")
    #         for index, (img, label) in enumerate(zip(images, labels)):
    #             print(index)
    #             img = img.unsqueeze(0).cuda()
    #             target_label = torch.tensor([label]).cuda()
    #             #deepfool_adv = attacker.generate("deepfool", img, target_label, overshoot=overshoot).cpu()
    #             deepfool_adv = DeepFool.forward()
    #             deepfool_adversarial_images.append(deepfool_adv)
    #             torch.cuda.empty_cache()

    #overshoots = [0.01, 0.02, 0.05, 0.1]
    #overshoots = [0.0001, 0.001, 0.005]
    iters = [1, 5, 10, 15, 20, 100]
    overshoots = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    for iter in iters:
        for overshoot in overshoots:
            defu = DeepFool(model, overshoot, 'cuda', steps=iter)
            deepfool_adversarial_images = []
            for i, (images, labels) in enumerate(testloader):
                print(f"{i}")
                #images = images.unsqueeze(0).cuda()
                images, labels = images.cuda(), labels.cuda()
                #target_labels = torch.tensor([labels]).cuda()
                #deepfool_adv = attacker.generate("deepfool", img, target_label, overshoot=overshoot).cpu()
                deepfool_adv = defu.forward(images, labels)
                deepfool_adversarial_images.append(deepfool_adv)
                torch.cuda.empty_cache()

            deepfool_adversarial_images = torch.cat(deepfool_adversarial_images, 0)
            torch.save(deepfool_adversarial_images, f"./adversarial_example/cnn/{iter}_deepfool_adversarial_cnn_{overshoot}.pth")
    #c_values = [1e-4, 1e-3, 1e-2, 1e-1]
    # c_values = [5e-1]
    # for c_value in c_values:
    #     cw_adversarial_images = []
    #     for i, (inputs, labels) in enumerate(testloader):
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #         print(f"Training progress: {i+1}/{total_train_batches}")
    #         cw_adv = attacker.generate("c&w", inputs, labels, c=c_value).cpu()
    #         cw_adversarial_images.append(cw_adv)
    #     cw_adversarial_images = torch.cat(cw_adversarial_images, 0)
    #     torch.save(cw_adversarial_images, f"./adversarial_example/cnn/cw_adversarial_cnn_{c_value}.pth")
    # epsilons=[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    # for epsilon in epsilons:
    #     print(f"Generating adversarial examples for epsilon: {epsilon}")
    #     fgsm_adversarial_images = []
    #     pgd_adversarial_images = []
    #     pgd_half_adversarial_images = []
    #     for i, (inputs, labels) in enumerate(testloader):
    #         inputs, labels = inputs.cuda(), labels.cuda()
    #         print(f"Training progress: {i+1}/{total_train_batches}")
    #         fgsm_adv = attacker.generate("fgsm", inputs, labels, epsilon=epsilon).cpu()
    #         #pgd_adv_half = attacker.generate("pgd", inputs, labels, epsilon=epsilon, alpha=0.01, iters=20).cpu()
    #         #pgd_adv = attacker.generate("pgd", inputs, labels, epsilon=epsilon, alpha=0.01, iters=40).cpu()
    #         fgsm_adversarial_images.append(fgsm_adv)
    #         #pgd_adversarial_images.append(pgd_adv)
    #         #pgd_half_adversarial_images.append(pgd_adv_half)

    #     fgsm_adversarial_images = torch.cat(fgsm_adversarial_images, 0)
    #     #pgd_adversarial_images = torch.cat(pgd_adversarial_images, 0)
    #     #pgd_half_adversarial_images = torch.cat(pgd_half_adversarial_images, 0)
    #     torch.save(fgsm_adversarial_images, f"./adversarial_example/cnn/fgsm_adversarial_cnn_{epsilon}.pth")
    #     #torch.save(pgd_adversarial_images, f"./adversarial_example/cnn/mini_pgd_adversarial_cnn_{epsilon}.pth")
    #     #torch.save(pgd_half_adversarial_images, f"./adversarial_example/cnn/mini_pgd_adversarial_half_cnn_{epsilon}.pth")
    #     #attacker.generate_and_save("boundary", images, labels, f"./adversarial_example/boundary_adversarial_{epsilon}.pth", epsilon=epsilon)


    # testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=6)
    # total_train_batches = len(testloader)
    # deepfool_adversarial_images = []
    # for i, (inputs, labels) in enumerate(testloader):
    #     print(f"Training progress: {i+1}/{total_train_batches}")
    #     inputs, labels = inputs.cuda(), labels.cuda()
    #     deepfool_adv = attacker.generate("deepfool", inputs, labels).cpu()
    #     deepfool_adversarial_images.append(deepfool_adv)
    
    # deepfool_adversarial_images = torch.cat(deepfool_adversarial_images, 0)
    # torch.save(deepfool_adversarial_images, f"./adversarial_example/deepfool_adversarial.pth")

        
    #attacker.generate_and_save("jsma", images, labels, f"./adversarial_example/jsma_adversarial.pth")
    # for epsilon in epsilons:
    #   attacker.generate_and_save("boundary", images, labels, f"./adversarial_example/boundary_adversarial_{epsilon}.pth", epsilon=epsilon)

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
