import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, Resize
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch import EfficientNet
def main():
    # GPU 캐시 정리
    torch.cuda.empty_cache()

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_WORKERS = 3
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations for the train set
    transform = Compose([
        #Resize((600, 600)),
        RandomResizedCrop(380),  # EfficientNet-B4's input size
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization parameters from ImageNet
    ])

    # Load the CIFAR-10 train data
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)

    # Load the CIFAR-10 test data
    test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    #set model
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
    scaler = GradScaler(growth_factor=1.5)

    # Train the model
    epochs = [30, 50]
    result = []
    for i in range(1):
        #if i == 0:
            #name = "sgdW"
            #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        if i == 0:
            name = "adamW"
            optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)
            #optimizer = optim.Adam(model.parameters(), lr=0.0002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                       factor=0.7, patience=3, verbose=True, 
                                                       threshold=0.001, threshold_mode='rel', 
                                                       cooldown=0, min_lr=1e-10, eps=1e-08)
        criterion = nn.CrossEntropyLoss()
        for iter in epochs:
            model.train().eval().to(device)
            for epoch in range(iter):  # loop over the dataset multiple times
                running_loss = 0.0
                correct_epoch = 0
                total_epoch = 0
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    if i % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.zero_grad()

                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) / GRADIENT_ACCUMULATION_STEPS

                    scaler.scale(loss).backward()  # Scale the loss using GradScaler

                    if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer)  # Perform the optimizer step
                        scaler.update()

                    # calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_epoch += labels.size(0)
                    correct_epoch += (predicted == labels).sum().item()

                    # print statistics
                    running_loss += loss.item()
                    if i % 100 == 99:    # print every 200 mini-batches
                        print('[%d, %5d] loss: %.3f \nAccuracy: %d %%' %
                            (epoch + 1, i + 1, running_loss / 100, (100 * correct_epoch / total_epoch)))
                        running_loss = 0.0
            correct = 0
            total = 0
            model.eval().to(device)
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            scheduler.step(accuracy)
            print(f"model:{name}, epoch:{iter}, Accuracy:{100 * accuracy}")
            result.append(f"model:{name}, epoch:{iter}, Accuracy:{100 * accuracy}")
            torch.save(model.state_dict(), './model_weights/'+
                    f'EfficientB4_ResizeCrop_{name}_{iter}_{100 * accuracy}.pth')

    print(result)

    print('Finished Training')

if __name__ == '__main__':
    main()