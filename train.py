from sklearn.metrics import balanced_accuracy_score
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms # type: ignore
from torch.utils.data import DataLoader
import os



# 0. Configuration

# Use the "split_dataset" folder created by running prep.py
DATA_DIR = "split_dataset"
SEED = 67
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Force PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable the benchmark mode (which picks the fastest algo, introducing noise)
    torch.backends.cudnn.benchmark = False 
    
    print(f"Random seed set to {seed}")

set_seed(SEED)



# 1. Data Transformations

# Use standard ImageNet normalization
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}



# 2. Load Data
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: DataLoader[tuple[torch.Tensor, torch.Tensor]](
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    for x in ["train", "val"]
}

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ["train", "val"]
}

class_names = image_datasets["train"].classes
num_classes = len(class_names)

print(f"Detected {num_classes} classes: {class_names}")



# 3. Set up ResNet with Transfer Learning

# Load ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze ALL parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the 'fc' (Fully Connected) layer - The Classifier
for param in model.fc.parameters():
    param.requires_grad = True

# Unfreeze 'layer4' - The last block of Convolutional layers
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(DEVICE)



# 4. Loss and Optimizer

# A. Calculate Class Weights
# Get all labels from the training set
train_targets = image_datasets['train'].targets 

# Count number of images per class
class_counts = np.bincount(train_targets)

# Calculate weights: Total_Samples / (Num_Classes * Class_Count)
total_samples = len(train_targets)
class_weights = torch.tensor(
    [total_samples / (num_classes * count) for count in class_counts],
    dtype=torch.float,
).to(DEVICE)

print(f"Class Counts: {class_counts}")
print(f"Using Class Weights: {class_weights}")

# B. Define Loss with Weights
# Pass the calculated weights so the model pays more attention to rare classes
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Only optimize parameters of the final layers
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    # momentum=0.9,
    weight_decay=1e-4,
)

# Decay LR
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=5,
)



# 5. Training Loop
def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int = 10
) -> nn.Module:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = torch.tensor(0, device=DEVICE)
            
            all_preds = list()
            all_labels = list()

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass if training
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = balanced_accuracy_score(all_labels, all_preds)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == "val":
                scheduler.step(epoch_loss)

    return model



# 6. Run Training
trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, NUM_EPOCHS)



# 7. Save the Model
model_pth = "skin_disease_resnet.pth"

torch.save(trained_model.state_dict(), model_pth)
print(f"Model saved successfully in {model_pth!r}.")
