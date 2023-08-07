import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, noise_size=100, classes=10):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.classes = classes
        self.embed = nn.Embedding(classes, noise_size)
        self.fc = nn.Linear(noise_size, 2048)
        self.bn1 = nn.BatchNorm2d(512)  # Changed from nn.BatchNorm1d(512)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, noise, labels):
        embed = self.embed(labels).view(-1, self.noise_size)
        x = torch.mul(noise, embed)
        x = F.leaky_relu(self.fc(x)).view(-1, 512, 2, 2)
        x = F.leaky_relu(self.bn2(self.conv1(self.bn1(x))))
        x = F.leaky_relu(self.bn3(self.conv2(x)))
        x = F.leaky_relu(self.bn4(self.conv3(x)))
        img = torch.tanh(self.conv4(x))
        return img
    
class Discriminator(nn.Module):
    def __init__(self, classes=10):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.embed = nn.Embedding(classes, 2048)  # Changed from nn.Embedding(classes, 100)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(2048, 1)

    def forward(self, img, labels):
        x = F.leaky_relu(self.bn1(self.conv1(img)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        embed = self.embed(labels)
        validity = torch.sigmoid(self.fc(x * embed.view(x.shape[0], -1)))
        return validity

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def calculate_fid_score(path_to_real_images, path_to_fake_images):
#     fid = fid_score.calculate_fid_given_paths([path_to_real_images, path_to_fake_images], 
#                                               batch_size=64, 
#                                               device=device)
#     print('FID score:', fid)
#     return fid

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_generator = Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_discriminator = Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

loss = nn.BCELoss()

dataset = CIFAR10(root='./data', download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


num_epochs = 200
for epoch in range(num_epochs):
    best_score = 10000000
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        noise = torch.randn(imgs.size(0), 100).to(device)
        gen_labels = torch.randint(0, 10, (imgs.size(0),)).to(device)
        gen_imgs = generator(noise, gen_labels)

        optimizer_generator.zero_grad()
        g_loss = loss(discriminator(gen_imgs, gen_labels), real)
        g_loss.backward()
        optimizer_generator.step()

        optimizer_discriminator.zero_grad()
        real_loss = loss(discriminator(real_imgs, labels), real)
        fake_loss = loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_discriminator.step()

        if i%100 == 99:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    # Compute Inception Score

    if score < best_score:
        torch.save(generator.state_dict(), 'generator_best.pth')
        best_score = score

    save_image(gen_imgs.data, f"gan_images/{epoch}.png")

torch.save(generator.state_dict(), 'generator_last.pth')