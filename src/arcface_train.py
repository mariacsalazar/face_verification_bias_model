import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import math


def get_accuracy(preds, y):
    """
    Compute the accuracy
    """
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


class ArcFaceLoss(nn.Module):
    def __init__(self, s=16.0, margin=0.3):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        # Clamp logits to [-1, 1] to avoid invalid arccos values
        logits = torch.clamp(logits, -1.0, 1.0)
        target_logit = torch.clamp(target_logit, -1.0, 1.0)

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits

def initialize_model(num_classes, learning_rate):
    model = models.resnet18(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Replace CrossEntropyLoss with ArcFaceLoss
    arcface_loss = ArcFaceLoss(s=16.0, margin=0.3)

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
        logits = model(X)  # Model outputs logits
        
        # Normalize the logits
        normalized_logits = nn.functional.normalize(logits, dim=1)
        
        # Apply ArcFace loss adjustment
        adjusted_logits = arcface_loss(normalized_logits, y)  
        
        # Compute CrossEntropyLoss with the adjusted logits
        loss = nn.CrossEntropyLoss()(adjusted_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = get_accuracy(adjusted_logits, y)
        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy

        # Print results per batch
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

            logits = model(X_test)  # Model outputs logits
            
            # Normalize the logits
            normalized_logits = nn.functional.normalize(logits, dim=1)
            
            # Apply ArcFace adjustment
            adjusted_logits = arcface_loss(normalized_logits, y_test)
            
            # Compute CrossEntropyLoss with the adjusted logits
            loss_test = nn.CrossEntropyLoss()(adjusted_logits, y_test)

            accuracy_test = get_accuracy(adjusted_logits, y_test)

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
    
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    for num_epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {num_epoch}/{num_epochs} ---")
        
        avg_train_loss, avg_train_accuracy = train_one_epoch(num_epoch, model, train_loader, optimizer, arcface_loss, device)
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        if num_epoch % checkpoint_interval == 0:
            checkpoint_path = f'{datestring}/checkpoint_epoch_{num_epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved as {checkpoint_path}')

        if num_epoch % test_interval == 0 or num_epoch == num_epochs:
            avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, arcface_loss, device)
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
    os.chdir(dname)

    train_and_save_model(
        num_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        num_workers=4,
        checkpoint_interval=1,
        test_interval=2,
        train_folder='data/imgs_subset/train',
        test_folder='data/imgs_subset/test'
    )


if __name__ == "__main__":
    main()
