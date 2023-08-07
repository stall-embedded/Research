import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from densenet import densenet_cifar

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations for the train set
transform = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 train data
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Load the CIFAR-10 test data
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the loss function and optimizer

# Train the model
epochs = [10, 20, 30, 100, 300, 500, 1000]
result = []
for i in range(3):
    model = densenet_cifar().to(device)
    if i == 0:
        name = "sgd"
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif i == 1:
        name = "adam1"
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
    else:
        name = "adam2"
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    for iter in epochs:
        for epoch in range(iter):  # loop over the dataset multiple times
            running_loss = 0.0
            correct_epoch = 0
            total_epoch = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                 # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_epoch += labels.size(0)
                correct_epoch += (predicted == labels).sum().item()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99 or (iter|i|epoch == 10):    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f \nAccuracy: %d %%' %
                        (epoch + 1, i + 1, running_loss / 100, (100 * correct_epoch / total_epoch)))
                    running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"model:{name}, epoch:{iter}, Accuracy:{100 * correct / total}")
        result.append(f"model:{name}, epoch:{iter}, Accuracy:{100 * correct / total}")
        torch.save(model.state_dict(), './model_weights/'+
                   f'densenet121_{name}_{iter}_{100 * correct / total}.pth')

print(result)

print('Finished Training')