import torch
import torch.nn as nn

class ActivationFunc(nn.Module):
    def __init__(self, alpha=1.0):
        super(ActivationFunc, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.atan(self.alpha * x)
    
class BasicModel_CNN(nn.Module):
    def __init__(
        self, num_channels, lr, optimizer, alpha=4.0, leak=0.01
    ):
        super().__init__()
        self.lr = lr
        self.leak = leak
        self.alpha = alpha

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.features = nn.Sequential(
            # preparation
            nn.Conv2d(num_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            # block 1
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            # block 2
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            # block 3
            nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            # classification
            nn.Linear(4096, 10, bias=False),
        )

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.gradients = None
        self.activations = None

        self.features[10].register_forward_hook(self.forward_hook)
        self.features[10].register_full_backward_hook(self.backward_hook)

    def forward(self, x):
        return self.features(x)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activations