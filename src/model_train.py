import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Initializing Hyperparameters
batch_size = 100         
learning_rate = 0.001    
num_epochs = 10          
embedding_size = 512     
num_classes = 1001   # There are 10572 images in our folder (using a subset of 1001 imgs for now)          

# Defining Transformations and Dataset
def prepare_dataset(batch_size, num_workers=2):
    # Using the transformations common to CASIA dataset
    transform = transforms.Compose([
        transforms.Resize((112, 112)),   
        transforms.ToTensor(),           
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    train_dataset = datasets.ImageFolder(root='small_images', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader

# Defining ArcMarginProduct Layer for ArcFace loss
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.s = s  # Scaling factor
        self.m = m  # Margin parameter
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        # Normalize features and weights
        cosine = nn.functional.linear(nn.functional.normalize(x), nn.functional.normalize(self.weight))
        # Apply ArcFace margin
        theta = torch.acos(cosine.clamp(-1.0, 1.0))
        target_logit = torch.cos(theta + self.m)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Compute output with margin for the target class
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        output *= self.s  # Scale output

        return output

# Defining the ResNet18 Model
class ResNet18ArcFace(nn.Module):
    def __init__(self, num_classes, feature_dim=512, use_arcface=True):
        super(ResNet18ArcFace, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        # Replacing the final fully connected layer of resnet18 with a custom one
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        
        self.use_arcface = use_arcface
        if use_arcface:
            self.arc_margin = ArcMarginProduct(feature_dim, num_classes)
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        
        # If using ArcFace, apply the margin layer
        if self.use_arcface and labels is not None:
            return self.arc_margin(features, labels)
        return self.classifier(features)

# Defining the Training Function
def train_model(model, train_loader, num_epochs, learning_rate, device, use_arcface):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    all_losses = []
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0  

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  
            optimizer.zero_grad()

            # Forward pass
            if use_arcface:
                outputs = model(images, labels)  
            else:
                outputs = model(images)  
            
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass and optimization
            loss.backward()        
            optimizer.step()       

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training Finished")
    
    plt.plot(all_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = prepare_dataset(batch_size)

    # Set use_arcface to 1 for ArcFace loss, 0 for simple loss (I am more confident about simple loss for now but both work lol)
    use_arcface = 0  

    model = ResNet18ArcFace(num_classes=num_classes, use_arcface=bool(use_arcface)).to(device)
    train_model(model, train_loader, num_epochs, learning_rate, device, use_arcface=bool(use_arcface))