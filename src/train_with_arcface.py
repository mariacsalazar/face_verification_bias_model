from iresnet import iresnet18  
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

class IResNetModified(nn.Module):
    def __init__(self):
        super(IResNetModified, self).__init__()
        self.base_model = iresnet18(pretrained=False, progress=True) 
        # Remove the fully connected layer for a 512-dimensional embedding
        self.base_model.fc = nn.Identity()
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.prelu(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.bn2(x)

        x = self.base_model.avgpool(x)  # Output shape (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to shape (batch_size, 512)
        x = self.base_model.features(x)
        return x

class ArcFace(torch.nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings, labels, W):
        # Normalize embeddings and the classifier weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(W, p=2, dim=0)

        # Transpose W to align dimensions for matrix multiplication
        cosine = torch.matmul(embeddings, W.t())  # Now cosine has shape [batch_size, num_classes]

        # Compute sine and phi for ArcFace margin
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply margin for the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.s
        return logits

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
    model = IResNetModified()  

    # Add classifier weight matrix W for ArcFace
    model.classifier = nn.Linear(512, num_classes, bias=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    arcface_loss = ArcFace(s=64.0, margin=0.5)  # Initialize ArcFace loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    arcface_loss = arcface_loss.to(device)
    
    print("Initialized model with ArcFace loss")
    return model, optimizer, arcface_loss, device

def train_one_epoch(epoch, model, train_loader, optimizer, arcface_loss, device):
    model.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        embeddings = model(X)  # Get 512-dimensional embeddings from the model
        W = model.classifier.weight  # Use the learnable weights as W

        # Apply ArcFace loss to generate logits
        logits = arcface_loss(embeddings, y, W)
        loss = nn.CrossEntropyLoss()(logits, y) 

        loss.backward()
        optimizer.step()

        # Compute accuracy
        accuracy = get_accuracy(logits, y)
        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy

        print(f"Epoch {epoch}, Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_accuracy = epoch_train_acc / len(train_loader)
    
    return avg_train_loss, avg_train_accuracy

def evaluate_model(model, test_loader, arcface_loss, device):
    model.eval()
    epoch_test_loss, epoch_test_acc = 0.0, 0.0
    total_test_samples = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            embeddings = model(X_test)
            W = model.classifier.weight  
            
            # Apply ArcFace loss to get logits
            logits = arcface_loss(embeddings, y_test, W)
            loss_test = nn.CrossEntropyLoss()(logits, y_test)  

            accuracy_test = get_accuracy(logits, y_test)

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
    model, optimizer, arcface_loss, device = initialize_model(num_classes, learning_rate)
    
    for num_epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {num_epoch}/{num_epochs} ---")
        avg_train_loss, avg_train_accuracy = train_one_epoch(num_epoch, model, train_loader, optimizer, arcface_loss, device)
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')

        if num_epoch % checkpoint_interval == 0:
            checkpoint_path = f'{datestring}/checkpoint_epoch_{num_epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved as {checkpoint_path}')

        if num_epoch % test_interval == 0 or num_epoch == num_epochs:
            avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, arcface_loss, device)
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

    checkpoint_path = f'{datestring}/final_model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print('Final model saved as "final_model.pt"')

def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    print(dname)
    os.chdir(dname)

    train_and_save_model(
        num_epochs=100, 
        batch_size=64, 
        learning_rate=0.001, 
        num_workers=4, 
        checkpoint_interval=1, 
        test_interval=2, 
        train_folder='data/imgs/train', 
        test_folder='data/imgs/test'
    )

if __name__ == "__main__":
    main()
