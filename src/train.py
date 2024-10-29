import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BiasModel():
    def __init__(self, num_classes):
        self.model_embeddings, self.model_classifier, self.model_bias = self._initialize_models(num_classes)

    def _initialize_models(self, num_classes):
        model_embeddings = models.resnet18(num_classes=num_classes)
        model_embeddings.fc = Identity()
        model_classifier = nn.Sequential(
            nn.Linear(512, num_classes, bias=True)
        )
        model_bias = nn.Sequential(
            nn.Linear(512, 1, bias=True)
        )
        return model_embeddings, model_classifier, model_bias
    

    def get_models(self):
        return self.model_embeddings, self.model_classifier, self.model_bias
    
    def get_parameters(self):
        return list(self.model_embeddings.parameters()) \
            + list(self.model_classifier.parameters()) \
            + list(self.model_bias.parameters())
    
    def train(self):
        self.model_embeddings.train()
        self.model_classifier.train()
        self.model_bias.train()
    
    def eval(self):
        self.model_embeddings.eval()
        self.model_classifier.eval()
        self.model_bias.eval()
    
    def save(self, path):
        torch.save(self.model_bias.state_dict(), path + '_bias.pt')
        torch.save(self.model_classifier.state_dict(), path + '_classifier.pt')
        torch.save(self.model_embeddings.state_dict(), path + '_embeddings.pt')
    
    def to(self, device):
        self.model_embeddings.to(device)
        self.model_classifier.to(device)
        self.model_bias.to(device)
    

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

def initialize_model(num_classes, learning_rate):
    """
    Initializes the ResNet18 model with the specified number of 
    output classes, along with the Adam optimizer and the cross-entropy loss
    function. It also detects whether a GPU (CUDA) is available and moves the 
    model and loss function to the appropriate device (GPU or CPU).

    """
      
    model = BiasModel(num_classes)
    optimizer = optim.Adam(model.get_parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn = loss_fn.to(device)
    print("initiated model")
    return model, optimizer, loss_fn, device
    
def train_one_epoch(epoch,model, train_loader, optimizer, loss_fn, device):
    
    """
    Train the model for one epoch. It iterates over the
    mini-batches in the training data, performs forward and backward passes,
    updates the model weights using the optimizer, and calculates the loss and
    accuracy for each batch. It returns the average loss and accuracy over the entire epoch.
    """

    model.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        embeddings = model.model_embeddings(X)
        train_preds = model.model_classifier(embeddings)
        forward_pass_loss = loss_fn(train_preds, y)
        predicted_loss = model.model_bias(embeddings)
        prediction_loss = torch.sum((predicted_loss - forward_pass_loss.item())**2)
        
        # TODO add a lambda here
        total_loss = forward_pass_loss + 0.01 * prediction_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute accuracy
        accuracy = get_accuracy(train_preds, y)
        epoch_train_loss += total_loss.item()
        epoch_train_acc += accuracy

        # Print results per batch
        print(f"Epoch {epoch},Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.4f}")

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_train_accuracy = epoch_train_acc / len(train_loader)
    
    return avg_train_loss, avg_train_accuracy

def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluate the trained model on the test dataset. It calculates
    the loss and accuracy for each mini-batch of the test data but 
    without updating the model weights (no gradient calculation). 
    It returns the average loss and accuracy for the entire test dataset.

    """

    model.eval()
    epoch_test_loss, epoch_test_acc = 0.0, 0.0
    total_test_samples = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_embeddings = model.model_embeddings(X_test)
            test_preds = model.model_classifier(test_embeddings)
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
        
    
        avg_train_loss, avg_train_accuracy = train_one_epoch(num_epoch,model, train_loader, optimizer, loss_fn, device)
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)


           
        if num_epoch % checkpoint_interval == 0:
            checkpoint_path = f'{datestring}/checkpoint_epoch_{num_epoch}'
            model.save(checkpoint_path)
            print(f'Checkpoint saved as {checkpoint_path}')

    
        if num_epoch % test_interval == 0 or num_epoch == num_epochs:
            avg_test_loss, avg_test_accuracy = evaluate_model(model, test_loader, loss_fn, device)
            test_loss.append(avg_test_loss)
            test_accuracy.append(avg_test_accuracy)
            print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}')

    checkpoint_path = f'{datestring}/final_model'
    model.save(checkpoint_path)
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
    test_folder='data/imgs_subset/train')



if __name__ == "__main__":
    main()
