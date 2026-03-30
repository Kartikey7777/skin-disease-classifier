import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ─── Config ───────────────────────────────────────────
DATASET_DIR = "data"
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.001
NUM_CLASSES = 7   # HAM10000 has 7 skin disease classes
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data Transforms ──────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Load Dataset ─────────────────────────────────────
train_data = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), train_transforms)
val_data   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   val_transforms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

# ─── Model (ResNet-50 Transfer Learning) ──────────────
model = models.resnet50(pretrained=True)

# Freeze all layers except last 3
for name, param in list(model.named_parameters())[:-6]:
    param.requires_grad = False

# Replace final layer for our classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ─── Loss & Optimizer ─────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ─── Training Loop ────────────────────────────────────
def train():
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100. * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), "model/best_model.pth")
            print(f"  ✅ Best model saved (Val Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    train()