import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# Create the generator and the discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define the optimizers
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Load the CIFAR-10 dataset
dataset = CIFAR10(root='./data', download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the gradient penalty
def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = Variable(interpolated, requires_grad=True)
    out = discriminator(interpolated)
    gradients = torch.autograd.grad(outputs=out, inputs=interpolated,
                                    grad_outputs=torch.ones(out.size(), device=device),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Train the WGAN
epochs = 100
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Move the images to the GPU
        real_images = real_images.to(device)

        # Train the discriminator
        optimizer_discriminator.zero_grad()
        real_pred = discriminator(real_images)
        real_error = -real_pred.mean()
        real_error.backward()

        noise = torch.randn(real_images.size(0), 100, 1, 1, device=device)
        fake_images = generator(noise)
        fake_pred = discriminator(fake_images.detach())
        fake_error = fake_pred.mean()
        fake_error.backward()

        gp = gradient_penalty(discriminator, real_images.data, fake_images.data)
        gp.backward()

        discriminator_error = real_error + fake_error + gp
        optimizer_discriminator.step()

        n_critic=5
        # Train the generator every n_critic iterations
        if i % n_critic == 0:
            optimizer_generator.zero_grad()
            fake_images = generator(noise)
            fake_pred = discriminator(fake_images)
            generator_error = -fake_pred.mean()
            generator_error.backward()
            optimizer_generator.step()

        # Print the losses and learning rates
        if i % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Batch {i}/{len(dataloader)}, "
                  f"Discriminator Error: {discriminator_error.item()}, "
                  f"Generator Error: {generator_error.item()}, "
                  f"Discriminator Learning Rate: {optimizer_discriminator.param_groups[0]['lr']}, "
                  f"Generator Learning Rate: {optimizer_generator.param_groups[0]['lr']}")

    # Generate images and save them
    noise = torch.randn(64, 100, 1, 1, device=device)
    generated_images = generator(noise)
    save_image(generated_images, f'gan_images/gan_image_{epoch}.png')