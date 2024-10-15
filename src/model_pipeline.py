import os
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
def sample_train_test(source_dir, train_dir, test_dir, train_ratio):
    """
    Split the imagen into train and test,
    keeping the structure of classes and save in separated folders
    
    Parameters:
    -----------
    source_dir: str
        Original folder
    train_dir: str
        Train folder
    test_dir: str
        Test folder
    train_ratio: float 
        Percentage in train of each class
    """
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)

 
        if os.path.isdir(class_dir):
            print(f"class name: {class_name}")

 
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

 
            all_files = os.listdir(class_dir)
            all_files = [f for f in all_files if os.path.isfile(os.path.join(class_dir, f))]  

 
            random.shuffle(all_files)

 
            split_idx = int(len(all_files) * train_ratio)
            train_files = all_files[:split_idx]
            test_files = all_files[split_idx:]

 
            for file_name in train_files:
                src_file = os.path.join(class_dir, file_name)
                dst_file = os.path.join(train_dir, class_name, file_name)
                shutil.copy2(src_file, dst_file)  

 
            for file_name in test_files:
                src_file = os.path.join(class_dir, file_name)
                dst_file = os.path.join(test_dir, class_name, file_name)
                shutil.copy2(src_file, dst_file)  

            print(f"Clase '{class_name}' processed: {len(train_files)} in train, {len(test_files)}  test.")


# source_dir = '../data/small_images'  # Carpeta original con subcarpetas (clases)
# train_dir = '../data/train'    # Carpeta de salida para el conjunto de entrenamiento
# test_dir = '../data/test'      # Carpeta de salida para el conjunto de prueba

# sample_train_test(source_dir, train_dir, test_dir, train_ratio=0.8)


def compute_accuracy(preds, y):
    """
    Compute the accuracy
    
    Parameters:
    -----------
    preds: Tensor
    y: Tensor
    
    Returns:
    --------
    accuracy: float
    """
    
    m = y.shape[0]
    hard_preds = torch.argmax(preds, dim=1)

    # Calculamos la precisión
    accuracy = torch.sum(hard_preds == y).item() / m
    return accuracy

def train_and_save_model(num_epochs=3, batch_size=100, learning_rate=0.001, num_workers=2, checkpoint_interval=2, test_interval=2):
    
    #Set restriccions ober checkpoints and test intervals.
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root='../data/train', transform=transform) 
    test_dataset = datasets.ImageFolder(root='../data/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Data loaded")

    num_classes = len(train_dataset.classes)
    
    # Model
    resnet18 = models.resnet18(num_classes=num_classes)
    optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18 = resnet18.to(device)
    loss_fn = loss_fn.to(device)

    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    # Train the model
    for num_epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {num_epoch}/{num_epochs} ---")
        
        resnet18.train()  
        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X, y = batch
            X, y = X.to(device), y.to(device)

            # Forward pass
            train_preds = resnet18(X)
            loss = loss_fn(train_preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            accuracy = compute_accuracy(train_preds, y)
            epoch_train_loss += loss.item()
            epoch_train_acc += accuracy

            # Print results per batch
            print(f"Epoch {num_epoch}/{num_epochs}, Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Save train metrics
        train_loss.append(epoch_train_loss / len(train_loader))
        train_accuracy.append(epoch_train_acc / len(train_loader))

        # Print train metrics
        print(f'Epoch [{num_epoch}/{num_epochs}] - Avg Train Loss: {train_loss[-1]:.4f}, Avg Train Accuracy: {train_accuracy[-1]:.4f}')

        # Save checkpoints
        if num_epoch % checkpoint_interval == 0:
            torch.save(resnet18.state_dict(), f'model_checkpoint_epoch_{num_epoch}.pt')
            print(f'Checkpoint guardado: modelo_epoch_{num_epoch}.pt')

        # Test
        if num_epoch % test_interval == 0 or num_epoch == num_epochs:
            resnet18.eval()
            epoch_test_loss, epoch_test_acc = 0.0, 0.0
            total_test_samples = 0
            with torch.no_grad():
                for batch_idx, (X_test, y_test) in enumerate(test_loader):
                    X_test, y_test = X_test.to(device), y_test.to(device)

                    test_preds = resnet18(X_test)
                    loss_test = loss_fn(test_preds, y_test)

                    
                    accuracy_test = compute_accuracy(test_preds, y_test)

                    
                    epoch_test_loss += loss_test.item() * X_test.size(0)
                    epoch_test_acc += accuracy_test * X_test.size(0)
                    total_test_samples += X_test.size(0)

            # Save test metrics
            test_loss.append(epoch_test_loss / total_test_samples)
            test_accuracy.append(epoch_test_acc / total_test_samples)

            print(f'Test Loss: {test_loss[-1]:.4f}, Test Accuracy: {test_accuracy[-1]:.4f}')

    # Save final model
    torch.save(resnet18.state_dict(), 'final_model.pt')
    print('Modelo final guardado en "final_model.pt"')

    
    return train_loss, train_accuracy, test_loss, test_accuracy


#train_loss, train_accuracy, test_loss, test_accuracy = train_and_save_model()

