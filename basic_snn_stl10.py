import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from spikingjelly.activation_based import neuron, layer, functional, surrogate
import cupy as cp

class BasicModel(nn.Module):
    def __init__(
        self, seq_num, num_channels, lr, optimizer, alpha=4.0, tau=2.0, leak=0.01
    ):
        super().__init__()
        self.lr = lr
        self.seq_num = seq_num
        self.leak = leak
        self.alpha = alpha
        self.tau = tau
        #SURROGATE = surrogate.Sigmoid(alpha=self.alpha)
        SURROGATE = surrogate.ATan(alpha=self.alpha)
        #SURROGATE = surrogate.LeakyKReLU(leak=self.leak)
        TAU = self.tau
        CUPY='cupy'

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.features = nn.Sequential(
            # preparation
            layer.Conv2d(num_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau = TAU, surrogate_function=SURROGATE, backend=CUPY),
            # block 1
            layer.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau = TAU, surrogate_function=SURROGATE, backend=CUPY),
            layer.AvgPool2d(2),
            # block 2
            layer.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau = TAU, surrogate_function=SURROGATE, backend=CUPY),
            layer.AvgPool2d(2),
            # block 3
            layer.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau = TAU, surrogate_function=SURROGATE, backend=CUPY),
            layer.AvgPool2d(2),
            layer.Flatten(),

            # classification
            layer.Linear(4096, 10, bias=False),
            neuron.LIFNode(tau=TAU, surrogate_function=SURROGATE, backend=CUPY),
        )

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.seq_num, 1, 1, 1, 1)
        x_seq = self.features(x_seq)
        y_hat = x_seq.mean(0)

        return y_hat