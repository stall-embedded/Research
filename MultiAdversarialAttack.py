import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
import copy
from torch.cuda.amp import autocast, GradScaler

class MultiAdversarialAttack:
    def __init__(self, model, device, is_SNN=False, kappa=0, const=1):
        self.model = model
        self.device = device
        self.is_SNN = is_SNN
        self.kappa = kappa
        self.const = const

        self.scaler = GradScaler()

    def get_gradient(self, inputs, target_label):
        inputs.requires_grad = True
        with autocast():
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, target_label)
        self.model.zero_grad()
        self.scaler.scale(loss).backward()
        functional.reset_net(self.model)
        return inputs.grad.data

    def fgsm(self, inputs, labels, epsilon):
        inputs.requires_grad = True
        with autocast():
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
        self.model.zero_grad()
        self.scaler.scale(loss).backward()
        perturbed_data = inputs + epsilon * inputs.grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        functional.reset_net(self.model)
        return perturbed_data

    def pgd(self, inputs, labels, epsilon, alpha, iters):
        perturbed_data = inputs.clone().detach().requires_grad_(True).to(self.device)
        criterion = nn.CrossEntropyLoss()
        for i in range(iters):
            print(f"PGD iter:{i}")
            with autocast():
                outputs = self.model(perturbed_data)
                loss = criterion(outputs, labels)
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            perturbed_data.data = perturbed_data + alpha * perturbed_data.grad.sign()
            perturbed_data.data = torch.clamp(perturbed_data, inputs - epsilon, inputs + epsilon)
            perturbed_data.data = torch.clamp(perturbed_data, 0, 1)
            perturbed_data = perturbed_data.detach().requires_grad_(True)
            functional.reset_net(self.model)
        return perturbed_data

    def deepfool(self, images, labels, num_classes=10, overshoot=0.02, max_iter=50):
        return self._deepfool_batch(images, labels, num_classes, overshoot, max_iter)

    def _deepfool_batch(self, images, labels, num_classes=10, overshoot=0.02, max_iter=50):
        images = images.clone().detach().to(self.device)
        original_labels = labels.clone().detach().to(self.device)
        r_tot = torch.zeros_like(images).to(self.device)

        for i in range(max_iter):
            print(f"Deepfool iter:{i}")
            outputs = self.model(images + r_tot)
            if (outputs.argmax(1) != original_labels).sum() == 0:
                break

            pert = float('inf')
            preds = outputs.argmax(1)

            for k in range(num_classes):
                if k == original_labels:
                    continue

                w_k = self.get_gradient(images + r_tot, k) - self.get_gradient(images + r_tot, original_labels)
                f_k = outputs[:, k] - outputs[torch.arange(images.size(0)), original_labels]
                pert_k = torch.abs(f_k) / torch.norm(w_k.view(images.size(0), -1), dim=1)

                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-4) * w / torch.norm(w.view(images.size(0), -1), dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            r_tot += r_i

        adv_images = images + (1 + overshoot) * r_tot
        return adv_images

    def carlini_wagner(self, images, labels, target=None, max_iter=500, learning_rate=0.01):
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size = images.size(0)
        one_hot_labels = torch.eye(len(self.model(images[0])))[labels].to(self.device)
        
        def f(x):
            with autocast():
                outputs = self.model(x)
                i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
                j = torch.masked_select(outputs, one_hot_labels.byte())
            return torch.clamp(i - j, min=-self.kappa)

        w = torch.zeros_like(images, requires_grad=True).to(self.device)
        optimizer = optim.Adam([w], lr=learning_rate)

        for step in range(max_iter):
            print(f"C&W iter:{step}")
            with autocast():
                new_imgs = torch.tanh(w) * 0.5 + 0.5
                loss = nn.MSELoss()(new_imgs, images) + torch.sum(self.const * f(new_imgs))
            optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=self.is_SNN)
            self.scaler.step(optimizer)
            self.scaler.update()
            functional.reset_net(self.model)

        adv_imgs = torch.tanh(w) * 0.5 + 0.5
        return adv_imgs
    
    def jsma_attack(self, images, original_labels, num_classes, theta=1, max_iter=40):
        images = images.to(self.device)
        original_labels = original_labels.to(self.device)
        batch_size = images.size(0)
        
        # 무작위 타겟 클래스 지정
        all_classes = torch.arange(num_classes).to(self.device)
        targets = torch.stack([torch.choice(all_classes[all_classes != lbl]) for lbl in original_labels])

        images = images.clone().detach().requires_grad_(True)
        current_labels = self.model(images).argmax(dim=1)

        finished_indices = (current_labels == targets)

        for i in range(max_iter):
            print(f"JSMA iter:{i}")
            outputs = self.model(images)
            current_labels = outputs.argmax(dim=1)
            finished_indices = (current_labels == targets)

            if finished_indices.all():
                break

            self.model.zero_grad()
            one_hot_targets = torch.eye(outputs.size(1))[targets].to(self.device)
            loss = torch.sum(outputs * one_hot_targets) 
            loss.backward(retain_graph=True)

            gradient = images.grad.data
            saliency_map = torch.mul((gradient > 0).float(), (images < 1).float()) - torch.mul((gradient < 0).float(), (images > 0).float())
            saliency_map = saliency_map.abs()

            _, indices = saliency_map.view(batch_size, -1).sort(dim=1, descending=True)
            pixel_to_perturb = indices[:, 0]

            perturbations = torch.zeros_like(images)
            for idx, pixel_idx in enumerate(pixel_to_perturb):
                perturbations[idx].view(-1)[pixel_idx] = theta

            images = torch.clamp(images + perturbations, 0, 1)
            images.grad.data.zero_()
            functional.reset_net(self.model)

        return images
    
    def boundary_attack(self, images, epsilon=0.1, max_iter=500, alpha=0.01):
        images = images.to(self.device)
        
        original_outputs = self.model(images)
        original_classes = original_outputs.argmax(dim=1)
        
        all_classes = torch.arange(original_outputs.size(1)).to(self.device)
        targets = torch.stack([torch.choice(all_classes[all_classes != oc]) for oc in original_classes])
        
        perturbed_images = images.clone().detach() + epsilon * torch.randn_like(images).sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        for i in range(max_iter):
            print(f"Boundary iter:{i}")
            perturbed_images.requires_grad_(True)
            with autocast():
                outputs = self.model(perturbed_images)

            perturbed_images = torch.where(outputs.argmax(dim=1)[:, None, None, None] == targets[:, None, None, None],
                                        images + (1 - alpha) * (perturbed_images - images),
                                        images + alpha * (perturbed_images - images))

            perturbed_images = torch.clamp(perturbed_images, 0, 1)
            functional.reset_net(self.model)

        return perturbed_images

    def generate_and_save(self, attack_type, inputs, labels, filename, **kwargs):
        print(f"Generating {attack_type} adversarial examples...")
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        if attack_type == "fgsm":
            perturbed_data = self.fgsm(inputs, labels, kwargs['epsilon'])
        elif attack_type == "pgd":
            perturbed_data = self.pgd(inputs, labels, kwargs['epsilon'], kwargs['alpha'], kwargs['iters'])
        elif attack_type == "deepfool":
            perturbed_data = self.deepfool(inputs, labels, **kwargs)
        elif attack_type in ["carlini_wagner", "c&w"]:
            perturbed_data = self.carlini_wagner(inputs, labels, **kwargs)
        elif attack_type == "jsma":
            perturbed_data = self.jsma_attack(inputs, labels, **kwargs)
        if attack_type == "boundary":
            perturbed_data = self.boundary_attack(inputs, labels, **kwargs)
        else:
            raise ValueError("Invalid attack type")

        torch.save(perturbed_data, filename)