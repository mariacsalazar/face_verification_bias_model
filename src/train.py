import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from iresnet import iresnet18
from losses import ArcFace
from partial_fc_v2 import PartialFC_V2

from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "/train_tmp/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

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
    Initializes the ResNet18 model with ArcFace for the specified number of 
    output classes, along with the Adam optimizer.
    """
    model =  iresnet18(pretrained=False, progress=True,num_features = num_classes) 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = ArcFace()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    # model.register_comm_hook(None, fp16_compress_hook)
    partial_fc = PartialFC_V2(loss_fn, 512, num_classes, sample_rate=0.2).to(device)

    print("Initiated Arcface model")
    return model, optimizer, partial_fc, device

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, device):

    """
    Train the model for one epoch. It iterates over the
    mini-batches in the training data, performs forward and backward passes,
    updates the model weights using the optimizer, and calculates the loss and
    accuracy for each batch. It returns the average loss and accuracy over the entire epoch.
    """
    model.train()
    loss_fn.train()
    epoch_train_loss, epoch_train_acc = 0.0, 0.0

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        if X.shape[0] != 64:
            continue
        # print(X.shape)

        # Forward pass
        optimizer.zero_grad()
        train_preds, embedding = model(X)
        # loss = loss_fn(train_preds, y)
        loss: torch.Tensor = loss_fn(embedding, y)

        
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
            if X_test.shape[0] != 64:
                continue

            X_test, y_test = X_test.to(device), y_test.to(device)

            test_preds, embedding = model(X_test)
            # loss_test = loss_fn(test_preds, y_test)

            loss_test: torch.Tensor = loss_fn(embedding, y_test)
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
        num_epochs=100, 
        batch_size=64, 
        learning_rate=0.001, 
        num_workers=12, 
        checkpoint_interval=1, 
        test_interval=1, 
        train_folder='data/imgs_subset/train', 
        test_folder='data/imgs_subset/test'
    )

if __name__ == "__main__":
    main()
