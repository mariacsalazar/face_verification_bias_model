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
from iresnet import iresnet18

# Using the Arcface implementation from Insightface
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        #logits = torch.clamp(logits, -1.0, 1.0)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()

        logits = logits * self.s   
        return logits

# Utilities
def get_accuracy(preds, y):
    """
    Compute the accuracy

    """
    m = y.shape[0]
    hard_preds = torch.argmax(preds, dim=1)
    accuracy = torch.sum(hard_preds == y).item() / m
    return accuracy

def load_data(train_folder, test_folder, batch_size, num_workers):
    """
    Loads and preprocesses the training and testing datasets
    from the specified folders. It applies transformations to resize the images,
    convert them to tensors, and normalize the pixel values.
     
    It returns DataLoader objects for both datasets to facilitate mini-batch 
    processing during training and evaluation.
    """
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

def initialize_model(num_classes, learning_rate, use_arcface):
    """
    Initializes the model, optimizer, loss function, etc.
    """
    if use_arcface:
        model = iresnet18(pretrained=False, progress=True) 
        # Add classifier weight matrix W for ArcFace
        model.classifier = nn.Linear(512, num_classes, bias=False)
        loss_fn = ArcFace(s=64.0, margin=0.5)
    else:
        model = iresnet18(pretrained=False, progress=True, num_features = num_classes)
        loss_fn = nn.CrossEntropyLoss()

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    print(f"Initialized model with {'ArcFace' if use_arcface else 'CrossEntropy'} loss")
    return model, optimizer, loss_fn, device

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, device, use_arcface):
    model.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(X)  # Always compute embeddings
        if use_arcface:
            # Normalize embeddings and classifier weights
            embeddings = F.normalize(embeddings, p=2, dim=1)
            model.classifier.weight.data = F.normalize(model.classifier.weight.data, p=2, dim=1)
            # Compute logits
            logits = F.linear(embeddings, model.classifier.weight)
            # Clamp logits to avoid out-of-bound values
            logits = torch.clamp(logits, -1.0, 1.0)
            # Use arcface loss adjustements
            logits = loss_fn(logits, y)
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            logits = embeddings  # Direct logits for CrossEntropy
            loss = loss_fn(logits, y)

        # Backward and optimize
        loss.backward()

        # Apply gradient clipping to stabilize training (use if needed)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Compute accuracy
        accuracy = get_accuracy(logits, y)
        epoch_train_loss += loss.item()
        epoch_train_acc += accuracy

        print(f"Epoch {epoch}, Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_accuracy = epoch_train_acc / len(train_loader)
    return avg_train_loss, avg_train_accuracy

def evaluate_model(model, test_loader, loss_fn, device, use_arcface):
    model.eval()
    epoch_test_loss, epoch_test_acc = 0.0, 0.0
    total_test_samples = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Forward pass
            embeddings = model(X_test)  # Always compute embeddings
            if use_arcface:
                logits = F.linear(embeddings, model.classifier.weight)  # Compute logits
                logits = loss_fn(logits, y_test)  # Apply ArcFace adjustments

                # Compute CrossEntropy loss
                loss_test = nn.CrossEntropyLoss()(logits, y_test)
            else:
                logits = embeddings  # Direct logits for CrossEntropy
                loss_test = loss_fn(logits, y_test)

            # Compute accuracy
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

# Main Training Loop
def train_and_save_model(num_epochs, batch_size, learning_rate, num_workers, checkpoint_interval, test_interval, train_folder, test_folder, use_arcface):
    """
    Main training loop with optional learning rate scheduler for ArcFace.
    """
    datestring = make_checkpoint_dir()
    train_loader, test_loader, num_classes = load_data(train_folder, test_folder, batch_size, num_workers)
    model, optimizer, loss_fn, device = initialize_model(num_classes, learning_rate, use_arcface)

    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        avg_train_loss, avg_train_accuracy = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, device, use_arcface)
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        if epoch % checkpoint_interval == 0:
            checkpoint_path = f'{datestring}/checkpoint_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved as {checkpoint_path}')

        if epoch % test_interval == 0 or epoch == num_epochs:
            avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, loss_fn, device, use_arcface)
            test_loss.append(avg_test_loss)
            test_accuracy.append(avg_test_accuracy)
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

    checkpoint_path = f'{datestring}/final_model.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print('Final model saved as "final_model.pt"')
    
def main():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(os.path.dirname(abspath))
    os.chdir(dname)

    use_arcface = True  # Toggle this for CrossEntropy or ArcFace
    train_and_save_model(
        num_epochs=100,
        batch_size=64,
        learning_rate=0.001,
        num_workers=4,
        checkpoint_interval=1,
        test_interval=2,
        train_folder='data/imgs_subset/train', 
        test_folder='data/imgs_subset/test',
        use_arcface=use_arcface
    )

if __name__ == "__main__":
    main()
