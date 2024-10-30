import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def get_accuracy(preds, y):
    m = y.shape[0]
    hard_preds = torch.argmax(preds, dim=1)
    accuracy = torch.sum(hard_preds == y).item() / m
    return accuracy

def load_data(train_folder, test_folder, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_dataset.classes)
    print("Data loaded")
    return train_loader, test_loader, num_classes


def initialize_model(num_classes, learning_rate):
    """
    Initializes the ResNet18 model with ArcFace for the specified number of 
    output classes, along with the Adam optimizer.
    """
    model = resnet18(num_classes=num_classes)  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    print("Initiated Arcface model")
    return model, optimizer, loss_fn, device

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        train_preds = model(X)
        loss = loss_fn(train_preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        accuracy = get_accuracy(train_preds, y)
        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy

        # Print results per batch
        print(f"Epoch {epoch}, Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_accuracy = epoch_train_acc / len(train_loader)
    
    return avg_train_loss, avg_train_accuracy

def evaluate_model(model, test_loader, loss_fn, device):
    model.eval()
    epoch_test_loss, epoch_test_acc = 0.0, 0.0
    total_test_samples = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_preds = model(X_test)
            loss_test = loss_fn(test_preds, y_test)

            accuracy_test = get_accuracy(test_preds, y_test)

            epoch_test_loss += loss_test.item() * X_test.size(0)
            epoch_test_acc += accuracy_test * X_test.size(0)
            total_test_samples += X_test.size(0)

    avg_test_loss = epoch_test_loss / total_test_samples
    avg_test_accuracy = epoch_test_acc / total_test_samples

    return avg_test_loss, avg_test_accuracy

def make_checkpoint_dir():
    today = datetime.now()
    datestring = today.strftime("checkpoint/checkpoint_%Y_%m_%d__%H_%M_%S")
    Path(datestring).mkdir(parents=True, exist_ok=True)
    return datestring

def train_and_save_model(num_epochs, batch_size, learning_rate, num_workers, checkpoint_interval, test_interval, train_folder, test_folder):
    datestring = make_checkpoint_dir()
    train_loader, test_loader, num_classes = load_data(train_folder, test_folder, batch_size, num_workers)
    model, optimizer, loss_fn, device = initialize_model(num_classes, learning_rate)
    
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    for num_epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {num_epoch}/{num_epochs} ---")
        
        avg_train_loss, avg_train_accuracy = train_one_epoch(num_epoch, model, train_loader, optimizer, loss_fn, device)
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        if num_epoch % checkpoint_interval == 0:
            checkpoint_path = f'{datestring}/checkpoint_epoch_{num_epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved as {checkpoint_path}')

        if num_epoch % test_interval == 0 or num_epoch == num_epochs:
            avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, loss_fn, device)
            test_loss.append(avg_test_loss)
            test_accuracy.append(avg_test_accuracy)
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

    checkpoint_path = f'{datestring}/final_model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print('Final model saved as "final_model.pt"')

    return train_loss, train_accuracy, test_loss, test_accuracy

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    print(dname)
    os.chdir(dname)

    train_and_save_model(
        num_epochs=10, 
        batch_size=64, 
        learning_rate=0.001, 
        num_workers=4, 
        checkpoint_interval=1, 
        test_interval=2, 
        train_folder='data/imgs_subset/train', 
        test_folder='data/imgs_subset/train'
    )

if __name__ == "__main__":
    main()
