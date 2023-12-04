import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.cuda.amp import autocast, GradScaler
import gc

class MultiAdversarialAttack:
    def __init__(self, model, device, is_SNN=False, kappa=0, const=1):
        self.model = model
        self.device = device
        self.is_SNN = is_SNN
        self.kappa = kappa
        self.const = const

        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss()

    def print_memory_usage(self):
        current_memory_allocated = torch.cuda.memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        usage_percentage = (current_memory_allocated / total_memory) * 100
        print(f"cm: {current_memory_allocated>>20}, tm: {total_memory>>20}")
        print(f"Memory Usage: {usage_percentage:.2f}%")

    # def fgsm(self, inputs, labels, epsilon):
    #     print("fgsm")
    #     inputs.requires_grad = True
    #     with autocast():
    #         outputs = self.model(inputs)
    #         loss = nn.CrossEntropyLoss()(outputs, labels)
    #     self.model.zero_grad()
    #     self.scaler.scale(loss).backward()
    #     perturbed_data = inputs + epsilon * inputs.grad.sign()
    #     perturbed_data = torch.clamp(perturbed_data, 0, 1)
    #     return perturbed_data
    def fgsm(self, inputs, labels, epsilon):
        print("fgsm")
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        perturbed_data = inputs + epsilon * inputs.grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data

    def pgd(self, inputs, labels, epsilon, alpha, iters):
        perturbed_data = inputs.clone().detach().requires_grad_(True).to(self.device)
        for i in range(iters):
            print(f"PGD iter:{i}")
            with autocast():
                outputs = self.model(perturbed_data)
                loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            perturbed_data.data = perturbed_data + alpha * perturbed_data.grad.sign()
            perturbed_data.data = torch.clamp(perturbed_data, inputs - epsilon, inputs + epsilon)
            perturbed_data.data = torch.clamp(perturbed_data, 0, 1)
            perturbed_data = perturbed_data.detach().requires_grad_(True)
        return perturbed_data
    
    
    # def get_gradient(self, inputs, target_label):
    #     inputs = inputs.clone().detach().requires_grad_(True).to(self.device)
    #     with autocast():
    #         outputs = self.model(inputs)
    #         loss = self.criterion(outputs, target_label)
    #     self.model.zero_grad()
    #     self.scaler.scale(loss).backward()
    #     gradient = inputs.grad.data.detach()
    #     return gradient
    def get_gradient(self, inputs, target_label):
        inputs = inputs.clone().detach().requires_grad_(True).to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, target_label)
        self.model.zero_grad()
        loss.backward()
        gradient = inputs.grad.data.detach()
        return gradient
    
    def deepfool(self, image, label, overshoot, num_classes=10, max_iter=40):
        image = image.to(self.device)
        original_label = label.unsqueeze(0).to(self.device)
        r_tot = torch.zeros_like(image).to(self.device)
        w = torch.zeros_like(image)
        for i in range(max_iter):
            print(f"Deepfool iter:{i}")
            self.print_memory_usage()
            with torch.enable_grad():
                outputs = self.model(image + r_tot)
            if outputs.argmax() != original_label:
                break

            pert = float('inf')
            preds = outputs.argmax()
            
            for k in range(num_classes):
                if k == original_label.item():
                    continue
                
                w_k = self.get_gradient(image + r_tot, torch.tensor([k], dtype=torch.long).to(self.device))-\
                self.get_gradient(image + r_tot, torch.tensor([original_label.item()], dtype=torch.long).to(self.device))
                f_k = outputs[0][k] - outputs[0][original_label.item()]
                pert_k = torch.abs(f_k.data) / torch.norm(w_k.data.flatten())

                if pert_k < pert:
                    pert = pert_k
                    w = w_k.data
                
                #del f_k, w_k
            r_i = (pert + 1e-4) * w.data / torch.norm(w.data.flatten())
            r_tot.add_(r_i.data)
            #del r_i

        adv_image = image + (1 + overshoot) * r_tot
        #del r_tot, image, original_label

        return adv_image

    def carlini_wagner(self, images, labels, c, targeted=False, kappa=0, max_iter=1000, learning_rate=0.01) :
        images = images.to(self.device)     
        labels = labels.to(self.device)

        def f(x) :
            with autocast():
                outputs = self.model(x)
                device = x.device
                one_hot_labels = torch.eye(len(outputs[0]), device=device)[labels].to(self.device)
                i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
                j = torch.masked_select(outputs, one_hot_labels.bool())
                
                if targeted :
                    return torch.clamp(i-j, min=-kappa)
                else :
                    return torch.clamp(j-i, min=-kappa)
            
        w = torch.zeros_like(images, requires_grad=True).to(self.device)
        optimizer = optim.Adam([w], lr=learning_rate)
        prev = 1e10
        
        for step in range(max_iter) :
            with autocast():
                a = 1/2*(nn.Tanh()(w) + 1)
                loss1 = nn.MSELoss(reduction='sum')(a, images)
                loss2 = torch.sum(c*f(a))
                cost = loss1 + loss2
            optimizer.zero_grad()
            self.scaler.scale(cost).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # Early Stop when loss does not converge.
            if step % (max_iter//10) == 0 :
                if cost > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                prev = cost
            
            print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')

        attack_images = 1/2*(nn.Tanh()(w) + 1)
        return attack_images
    
    def jsma_attack(self, images, original_labels, num_classes=10, theta=1, max_iter=40):
        images = images.to(self.device)
        original_labels = original_labels.to(self.device)
        batch_size = images.size(0)
        
        # 무작위 타겟 클래스 지정
        all_classes = torch.arange(num_classes).to(self.device)
        targets = torch.stack([all_classes[all_classes != lbl][torch.randint(0, len(all_classes[all_classes != lbl]), (1,))].item() for lbl in original_labels])

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

        return images
    
    def boundary_attack(self, images, epsilon, max_iter=500, alpha=0.01):
        images = images.to(self.device)
        
        original_outputs = self.model(images)
        original_classes = original_outputs.argmax(dim=1)
        
        all_classes = torch.arange(original_outputs.size(1)).to(self.device)
        targets = torch.stack([all_classes[all_classes != oc][torch.randint(0, len(all_classes[all_classes != oc]), (1,))] for oc in original_classes])
        
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

        return perturbed_images

    def generate(self, attack_type, input, label, **kwargs):
        input = input.cuda()
        label = label.cuda()
        if attack_type == "fgsm":
            perturbed_data = self.fgsm(input, label, kwargs['epsilon'])
        elif attack_type == "pgd":
            perturbed_data = self.pgd(input, label, kwargs['epsilon'], kwargs['alpha'], kwargs['iters'])
        elif attack_type == "deepfool":
            perturbed_data = self.deepfool(input, label, kwargs['overshoot'])
        elif attack_type in ["carlini_wagner", "c&w"]:
            perturbed_data = self.carlini_wagner(input, label, kwargs['c'])
        elif attack_type == "jsma":
            perturbed_data = self.jsma_attack(input, label, **kwargs)
        elif attack_type == "boundary":
            perturbed_data = self.boundary_attack(input, kwargs['epsilon'])
        else:
            raise ValueError("Invalid attack type")
        print(f"perturbed_data:{(perturbed_data.element_size() * perturbed_data.nelement())}Bytes")
        return perturbed_data