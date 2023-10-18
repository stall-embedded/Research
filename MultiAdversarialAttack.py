import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional
import copy

class MultiAdversarialAttack:
    def __init__(self, model, device, is_SNN=False, kappa=0, const=1):
        self.model = model
        self.device = device
        self.is_SNN = is_SNN
        self.kappa = kappa
        self.const = const

    def get_gradient(self, inputs, target_label):
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, target_label)
        self.model.zero_grad()
        loss.backward(retain_graph=self.is_SNN)
        functional.reset_net(self.model)
        return inputs.grad.data

    def fgsm(self, inputs, labels, epsilon):
        inputs.requires_grad = True
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.model.zero_grad()
        loss.backward(retain_graph = self.is_SNN)
        perturbed_data = inputs + epsilon * inputs.grad.sign()
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        # if self.is_SNN:
        #     functional.reset_net(self.model)
        functional.reset_net(self.model)
        return perturbed_data

    def pgd(self, inputs, labels, epsilon, alpha, iters):
        perturbed_data = inputs.clone().detach().requires_grad_(True).to(self.device)
        for _ in range(iters):
            outputs = self.model(perturbed_data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            self.model.zero_grad()
            loss.backward(retain_graph = self.is_SNN)
            new_data = perturbed_data + alpha * perturbed_data.grad.sign()
            new_data = new_data.detach().requires_grad_(True)
            perturbed_data = new_data
            perturbed_data = torch.clamp(perturbed_data, inputs - epsilon, inputs + epsilon)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)
            functional.reset_net(self.model)
        return perturbed_data

    def deepfool(self, images, labels, num_classes=10, overshoot=0.02, max_iter=50):
        adv_images = []
        for image, label in zip(images, labels):
            adv_image = self._deepfool_single(image.unsqueeze(0), label, num_classes, overshoot, max_iter)
            adv_images.append(adv_image)
        return torch.stack(adv_images)
    
    def _deepfool_single(self, image, label, num_classes=10, overshoot=0.02, max_iter=50):
        image = image.clone().detach().to(self.device)
        output = self.model(image)
        input_shape = image.shape
        w = torch.zeros(input_shape).to(self.device)
        r_tot = torch.zeros(input_shape).to(self.device)
        image = image.unsqueeze(0)
        original_label = label

        for _ in range(max_iter):
            output = self.model(image + r_tot)
            if output.argmax(1) != original_label:
                break

            pert = float('inf')
            pred = output[0].argmax()

            for k in range(num_classes):
                if k == original_label:
                    continue

                w_k = self.get_gradient(image + r_tot, k) - self.get_gradient(image + r_tot, original_label)
                f_k = output[0][k] - output[0][original_label]
                pert_k = abs(f_k) / torch.norm(w_k.flatten())

                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert + 1e-4) * w / torch.norm(w)
            r_tot += r_i
            functional.reset_net(self.model)

        adv_image = image + (1 + overshoot) * r_tot
        return adv_image.squeeze(0)

    def carlini_wagner(self, image, label, target=None, max_iter=1000, learning_rate=0.01):
        image = image.to(self.device)
        label = torch.tensor([label]).to(self.device)
        def f(x):
            outputs = self.model(x)
            one_hot_labels = torch.eye(len(outputs[0]))[label].to(self.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.byte())
            return torch.clamp(i - j, min=-self.kappa)

        w = torch.zeros_like(image, requires_grad=True).to(self.device)
        optimizer = optim.Adam([w], lr=learning_rate)

        for step in range(max_iter):
            new_img = torch.tanh(w) * 0.5 + 0.5
            loss = nn.MSELoss()(new_img, image) + torch.sum(self.const * f(new_img))
            optimizer.zero_grad()
            loss.backward(retain_graph = self.is_SNN)
            optimizer.step()
            functional.reset_net(self.model)

        adv_img = torch.tanh(w) * 0.5 + 0.5
        return adv_img.detach()
    
    def jsma_attack(self, image, target, theta=1, max_iter=100):
        """
        JSMA Attack
        :param image: Input image
        :param target: Target label
        :param theta: Perturbation introduced to modified components (can be positive or negative)
        :param max_iter: Maximum number of iterations
        :return: Adversarial example
        """
        image = image.to(self.device)
        target = torch.tensor([target]).to(self.device)
        image = image.clone().detach().requires_grad_(True)
        output = self.model(image)
        current_label = output.argmax().item()

        # If the image is already misclassified, no need to attack
        if current_label == target:
            return image

        for i in range(max_iter):
            output = self.model(image)
            if output.argmax().item() == target:
                break

            # Compute the gradient
            self.model.zero_grad()
            output[0, target].backward(retain_graph=True)
            gradient = image.grad.data

            # Compute the saliency map
            saliency_map = torch.mul((gradient > 0).float(), (image < 1).float()) - torch.mul((gradient < 0).float(), (image > 0).float())
            saliency_map = saliency_map.abs()

            # Find the pixel to perturb
            _, indices = saliency_map.view(-1).sort(descending=True)
            pixel_to_perturb = indices[0]

            # Perturb the pixel
            perturbation = torch.zeros_like(image)
            perturbation.view(-1)[pixel_to_perturb] = theta
            image = torch.clamp(image + perturbation, 0, 1)
            functional.reset_net(self.model)

        return image
    
    def boundary_attack(self, image, target, epsilon=0.1, max_iter=1000, alpha=0.01):
        """
        Boundary Attack
        :param image: Input image
        :param target: Target label
        :param epsilon: Maximum perturbation
        :param max_iter: Maximum number of iterations
        :param alpha: Step size
        :return: Adversarial example
        """
        image = image.to(self.device)
        target = torch.tensor([target]).to(self.device)
        perturbed_image = image.clone().detach() + epsilon * torch.randn_like(image).sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        for i in range(max_iter):
            perturbed_image.requires_grad_(True)
            output = self.model(perturbed_image)

            # If the perturbed image is misclassified, move it closer to the original image
            if output.argmax().item() == target:
                perturbed_image = image + (1 - alpha) * (perturbed_image - image)
            else:
                perturbed_image = image + alpha * (perturbed_image - image)

            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            functional.reset_net(self.model)

        return perturbed_image

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
            perturbed_data = []
            for img, lbl in zip(inputs, labels):
                perturbed_data.append(self.carlini_wagner(img.unsqueeze(0), lbl.item(), **kwargs))
            perturbed_data = torch.stack(perturbed_data)
        elif attack_type == "jsma":
            perturbed_data = []
            for img, lbl in zip(inputs, labels):
                perturbed_data.append(self.jsma_attack(img.unsqueeze(0), lbl.item(), **kwargs))
            perturbed_data = torch.stack(perturbed_data)
        elif attack_type == "boundary":
            perturbed_data = []
            for img, lbl in zip(inputs, labels):
                perturbed_data.append(self.boundary_attack(img.unsqueeze(0), lbl.item(), **kwargs))
            perturbed_data = torch.stack(perturbed_data)
        else:
            raise ValueError("Invalid attack type")

        torch.save(perturbed_data, filename)

# Example usage:
# attacker = AdversarialAttack(model, 'cuda')
# attacker.generate_and_save("fgsm", inputs, labels, "fgsm_adversarial.pth", epsilon=0.3)