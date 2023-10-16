import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from basic_snn_past import BasicModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from spikingjelly.activation_based import functional, neuron

BATCH_SIZE = 64
NUM_WORKERS = 6
ALPHA = 3.7132080089425044
TAU = 2.180830180029865

def main():
    torch.cuda.empty_cache()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_dataset = torch.load('cifar10_val_dataset.pth')
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    #testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    model = BasicModel(seq_num=50, num_channels=3, optimizer="Adam", lr=0.001, alpha=ALPHA, tau=TAU).cuda()
    functional.set_step_mode(model, step_mode='m')
    # for layer in model.features:
    #     if isinstance(layer, neuron.BaseNode):
    #         layer.backend = 'cupy'

    print("check1")
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model = None
    total_train_batches = len(trainloader)
    total_val_batches = len(valloader)
    scheduler = ReduceLROnPlateau(model.optimizer, 'max', factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(100):
        model.train()

        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            model.optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_batch = labels.size(0)
            correct_batch = (predicted == labels).sum().item()
            if torch.isnan(loss):
                print('Loss is NaN. Stopping training.')
                break

            epoch_train_loss += loss.item()
            epoch_train_correct += correct_batch
            epoch_train_total += total_batch

            acc = (100 * correct_batch / total_batch)
            print(f"Epoch {epoch+1} - Training progress: {i+1}/{total_train_batches} batches, Loss: {loss.item():.4f}, Acc: {acc}")
            functional.reset_net(model)
        
        train_losses.append(epoch_train_loss / total_train_batches)
        train_accuracies.append(epoch_train_correct / epoch_train_total)

        # 검증 루프
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        correct=0
        total=0
        with torch.no_grad():
            j = 0
            for inputs, labels in valloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print('Loss is NaN. Stopping Validating.')
                    break

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                print(f"Epoch {epoch+1} - Validation progress: {j+1}/{total_val_batches} batches, Loss: {loss.item():.4f}")
                j += 1
                functional.reset_net(model)

        val_loss /= len(valloader)
        val_acc = correct / total
        print(f"Epoch {epoch+1}, Val loss: {val_loss}, Val acc: {100*val_acc}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, 'best_basic_model_snn_sj.pth')
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    torch.save({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, 'training_logs.pth')

if __name__ == '__main__':
    main()
