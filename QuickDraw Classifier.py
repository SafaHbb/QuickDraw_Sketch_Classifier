!pip install torchmetrics scikit-learn lightning-utilities --quiet

# General libraries
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# PyTorch and TorchVision tools
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights

# Metrics from torchmetrics and sklearn
from torchmetrics import Accuracy, MeanMetric, Precision, Recall, F1Score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

"""##Check for GPU and Setup Logs"""

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Store model performance and training loss history
all_model_results = []
all_loss_histories = {}

"""##Load and Prepare QuickDraw Data

"""

# Define 5 selected classes for classification
selected_classes = ['apple', 'car', 'cat', 'dog', 'flower']
class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
samples_per_class = 600
data_path = "/content/quickdraw_dataset"

# Load .npy sketch files and assign labels
data_images, data_labels = [], []
for cls in selected_classes:
    path = os.path.join(data_path, cls + ".npy")
    data = np.load(path)[:samples_per_class]
    data_images.append(data)
    data_labels.extend([class_to_idx[cls]] * len(data))

# Combine and normalise images, convert labels to array
data_images = np.concatenate(data_images, axis=0).astype(np.float32) / 255.0
labels = np.array(data_labels)

"""##Create a Custom Dataset Class"""

# Converts numpy images into PIL images and applies transforms
class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].reshape(28, 28) * 255
        img = Image.fromarray(img.astype(np.uint8), mode='L').convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

"""##Define Image Transforms"""

# Data augmentation for training
train_transform = v2.Compose([
    v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=128, scale=(0.5, 1.0)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Resize and normalize for validation/test
eval_transform = v2.Compose([
    v2.Resize((128, 128)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""##Split Data and Create Dataloaders"""

# 80% train, 20% validation
full_dataset = QuickDrawDataset(data_images, labels, transform=None)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

# Create datasets with transforms
train_data = QuickDrawDataset(data_images[train_indices.indices], labels[train_indices.indices], transform=train_transform)
val_data = QuickDrawDataset(data_images[val_indices.indices], labels[val_indices.indices], transform=eval_transform)
test_data = QuickDrawDataset(data_images[val_indices.indices[:len(val_indices)//5]], labels[val_indices.indices[:len(val_indices)//5]], transform=eval_transform)

# Load datasets into PyTorch dataloaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

"""##Basic Training Loop (Loss + Accuracy)

"""

def train_one_epoch(model, dataloader, optimizer, criteria, num_classes):
    losses = MeanMetric().to(device)
    acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)

    model.train()
    for X, Y in tqdm(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criteria(preds, Y)
        loss.backward()
        optimizer.step()
        preds = preds.argmax(dim=1)
        losses.update(loss, X.size(0))
        acc.update(preds, Y)

    return losses.compute().item(), acc.compute().item()

"""##Basic Evaluation Function (Accuracy Only)"""

def eval_accuracy(model, dataloader, num_classes):
    acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    model.eval()

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=1)
            acc.update(preds, Y)

    return acc.compute().item()



"""##Extended Training Loop (All Metrics)"""

def train_one_epoch_extended(model, dataloader, optimizer, criteria, num_classes):
    losses = MeanMetric().to(device)
    acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    prec = Precision(task='multiclass', average='macro', num_classes=num_classes).to(device)
    rec = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
    f1 = F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)

    model.train()
    for X, Y in tqdm(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criteria(preds, Y)
        loss.backward()
        optimizer.step()
        preds = preds.argmax(dim=1)
        losses.update(loss, X.size(0))
        acc.update(preds, Y)
        prec.update(preds, Y)
        rec.update(preds, Y)
        f1.update(preds, Y)

    return {
        "loss": losses.compute().item(),
        "accuracy": acc.compute().item(),
        "precision": prec.compute().item(),
        "recall": rec.compute().item(),
        "f1": f1.compute().item()
    }

"""##Extended Evaluation (All Metrics)

"""

def eval_all_metrics(model, dataloader, num_classes):
    acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    prec = Precision(task='multiclass', average='macro', num_classes=num_classes).to(device)
    rec = Recall(task='multiclass', average='macro', num_classes=num_classes).to(device)
    f1 = F1Score(task='multiclass', average='macro', num_classes=num_classes).to(device)

    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=1)
            acc.update(preds, Y)
            prec.update(preds, Y)
            rec.update(preds, Y)
            f1.update(preds, Y)

    return {
        "accuracy": acc.compute().item(),
        "precision": prec.compute().item(),
        "recall": rec.compute().item(),
        "f1": f1.compute().item()
    }

"""##Plot ROC Curve for All Classes"""

def plot_multiclass_roc(model, dataloader, num_classes, class_names):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            outputs = model(X)
            y_true.extend(Y.cpu().numpy())
            y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_score = np.array(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

"""#**CNN**
This section defines and trains a simple Convolutional Neural Network (CNN) for classifying 5 Quick, Draw! sketch categories. It includes model definition, training, validation, evaluation, and metric tracking.

##Model Definition
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Define the Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

"""##Setup: Loss Function and Optimizer"""

# 2. Set up model, loss, optimizer
model = SimpleCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""##Training and Validation Loop"""

# 3. Training Loop with Validation Monitoring
epochs = 10
train_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # === Validation Evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(Y_batch.numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""##Visualise Training Progress"""

# 4. Plot Loss and Validation Accuracy
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("SimpleCNN Training")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.show()

"""##Test the Best Model"""

# 5. Load Best Model and Evaluate on Test Set
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(Y_batch.numpy())

"""##Print Test Metrics"""

# 6. Final Test Metrics
print("\n--- Final Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
print("Accuracy: ", accuracy_score(all_targets, all_preds))
print("Precision: ", precision_score(all_targets, all_preds, average='macro'))
print("Recall: ", recall_score(all_targets, all_preds, average='macro'))
print("F1 Score: ", f1_score(all_targets, all_preds, average='macro'))

"""##Save Model Results"""

all_loss_histories["CNN"] = train_losses  # === 7. Save Final Results ===
final_epoch = epochs
final_loss = train_losses[-1]
final_val_acc = val_accuracies[-1]
final_test_acc = accuracy_score(all_targets, all_preds)
final_precision = precision_score(all_targets, all_preds, average='macro')
final_recall = recall_score(all_targets, all_preds, average='macro')
final_f1 = f1_score(all_targets, all_preds, average='macro')

# Save the result as a dictionary
this_model_results = {
    "Model": "CNN",  # CHANGE this for each model: "SimpleCNN", "LeNet", etc.
    "Epochs": final_epoch,
    "Final Train Loss": round(final_loss, 4),
    "Val Accuracy": round(final_val_acc, 4),
    "Test Accuracy": round(final_test_acc, 4),
    "Precision": round(final_precision, 4),
    "Recall": round(final_recall, 4),
    "F1 Score": round(final_f1, 4)
}

# Add it to a global list (must define this list before running any model)
all_model_results.append(this_model_results)

"""#**LeNet**
This section builds and trains the LeNet model for sketch classification and evaluates its performance using multiple metrics.

##Model Definition
"""

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Define the Model
class LeNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # ← ReLU instead of tanh
        x = self.pool(torch.relu(self.conv2(x)))   # ← ReLU instead of tanh
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))                # ← ReLU instead of tanh
        x = torch.relu(self.fc2(x))                # ← ReLU instead of tanh
        return self.fc3(x)

"""##Model Setup"""

# 2. Instantiate model, loss, optimizer
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""##Training Loop"""

# 3. Training Loop with Validation Monitoring
epochs = 10
train_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # === Validation Evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(Y_batch.numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""##Plot Training Metrics"""

# 4. Plot Loss and Validation Accuracy
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("LeNet Training")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.show()

"""##Test the Model"""

# 5. Load Best Model and Evaluate on Test Set
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(Y_batch.numpy())

"""##Evaluate Performance"""

# 6. Final Test Evaluation
print("\n--- Final Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
print("Accuracy: ", accuracy_score(all_targets, all_preds))
print("Precision: ", precision_score(all_targets, all_preds, average='macro'))
print("Recall: ", recall_score(all_targets, all_preds, average='macro'))
print("F1 Score: ", f1_score(all_targets, all_preds, average='macro'))

all_loss_histories["LeNet"] = train_losses

"""##Save Results"""

# === 7. Save Final Results ===
final_epoch = epochs
final_loss = train_losses[-1]
final_val_acc = val_accuracies[-1]
final_test_acc = accuracy_score(all_targets, all_preds)
final_precision = precision_score(all_targets, all_preds, average='macro')
final_recall = recall_score(all_targets, all_preds, average='macro')
final_f1 = f1_score(all_targets, all_preds, average='macro')

# Save the result as a dictionary
this_model_results = {
    "Model": "LeNet",  # CHANGE this for each model: "SimpleCNN", "LeNet", etc.
    "Epochs": final_epoch,
    "Final Train Loss": round(final_loss, 4),
    "Val Accuracy": round(final_val_acc, 4),
    "Test Accuracy": round(final_test_acc, 4),
    "Precision": round(final_precision, 4),
    "Recall": round(final_recall, 4),
    "F1 Score": round(final_f1, 4)
}

# Add it to a global list (must define this list before running any model)
all_model_results.append(this_model_results)

"""#**ResNet18**
This section loads a pre-trained ResNet18 model, tunes it for Quick, Draw! classification, and evaluates its performance using accuracy, precision, recall, and F1 score.

##Model Definition
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Define and load ResNet18 model
def get_resnet_model(num_classes=5):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

"""##Model Setup"""

# 2. Set up model, loss function, optimizer
model = get_resnet_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""##Training and Validation Loop"""

# 3. Training with Validation Monitoring
epochs = 10
train_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # === Validation Evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(Y_batch.numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""##Plot Training Metrics"""

# 4. Plot Training Loss and Validation Accuracy
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("ResNet18 Training")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.show()

"""##Test the Model"""

# 5. Load Best Model and Evaluate on Test Set
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(Y_batch.numpy())

"""##Evaluate Performance"""

# 6. Final Test Metrics
print("\n--- Final Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
print("Accuracy: ", accuracy_score(all_targets, all_preds))
print("Precision: ", precision_score(all_targets, all_preds, average='macro'))
print("Recall: ", recall_score(all_targets, all_preds, average='macro'))
print("F1 Score: ", f1_score(all_targets, all_preds, average='macro'))

all_loss_histories["ResNet18"] = train_losses

"""##Save Results"""

# === 7. Save Final Results ===
final_epoch = epochs
final_loss = train_losses[-1]
final_val_acc = val_accuracies[-1]
final_test_acc = accuracy_score(all_targets, all_preds)
final_precision = precision_score(all_targets, all_preds, average='macro')
final_recall = recall_score(all_targets, all_preds, average='macro')
final_f1 = f1_score(all_targets, all_preds, average='macro')

# Save the result as a dictionary
this_model_results = {
    "Model": "ResNet",  # CHANGE this for each model: "SimpleCNN", "LeNet", etc.
    "Epochs": final_epoch,
    "Final Train Loss": round(final_loss, 4),
    "Val Accuracy": round(final_val_acc, 4),
    "Test Accuracy": round(final_test_acc, 4),
    "Precision": round(final_precision, 4),
    "Recall": round(final_recall, 4),
    "F1 Score": round(final_f1, 4)
}

# Add it to a global list (must define this list before running any model)
all_model_results.append(this_model_results)

"""#**AlexNet**
This section loads a pre-trained AlexNet model, tunes it for classifying Quick, Draw! sketches, and evaluates performance using standard classification metrics.

##Model Definition
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 1. Define and load AlexNet model
model = alexnet(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 5)
model = model.to(device)

"""##Model Setup"""

# 2. Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""##Training and Validation Loop"""

# 3. Training with Validation Monitoring
epochs = 10
train_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # === Validation Evaluation
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(Y_batch.numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""##Visualise Training Metrics"""

# 4. Plot Training Loss and Validation Accuracy
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("AlexNet Training")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.show()

"""##Test the Model"""

# 5. Load Best Model and Evaluate on Test Set
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(Y_batch.numpy())

"""##Evaluate Performance"""

# 6. Final Test Metrics
print("\n--- Final Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
print("Accuracy: ", accuracy_score(all_targets, all_preds))
print("Precision: ", precision_score(all_targets, all_preds, average='macro'))
print("Recall: ", recall_score(all_targets, all_preds, average='macro'))
print("F1 Score: ", f1_score(all_targets, all_preds, average='macro'))

all_loss_histories["AlexNet"] = train_losses

"""##Save Results"""

# === 7. Save Final Results ===
final_epoch = epochs
final_loss = train_losses[-1]
final_val_acc = val_accuracies[-1]
final_test_acc = accuracy_score(all_targets, all_preds)
final_precision = precision_score(all_targets, all_preds, average='macro')
final_recall = recall_score(all_targets, all_preds, average='macro')
final_f1 = f1_score(all_targets, all_preds, average='macro')

# Save the result as a dictionary
this_model_results = {
    "Model": "AlexNet",  # CHANGE this for each model: "SimpleCNN", "LeNet", etc.
    "Epochs": final_epoch,
    "Final Train Loss": round(final_loss, 4),
    "Val Accuracy": round(final_val_acc, 4),
    "Test Accuracy": round(final_test_acc, 4),
    "Precision": round(final_precision, 4),
    "Recall": round(final_recall, 4),
    "F1 Score": round(final_f1, 4)
}

# Add it to a global list (must define this list before running any model)
all_model_results.append(this_model_results)

"""#**ConvNeXt**
This section uses a pre-trained ConvNeXt model for sketch classification. It trains the model, validates accuracy, evaluates on the test set, and saves all final metrics.

##Model Definition
"""

# === ConvNeXt ===
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# 1. Load ConvNeXt model
def get_convnext_model(num_classes=5):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

model = get_convnext_model(num_classes=5).to(device)

"""##Loss and Optimizer"""

# 2. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

"""##Training and Validation Loop"""

# 3. Training Loop
epochs = 10
train_losses = []
val_accuracies = []
best_val_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, Y_batch in train_dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # === Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_targets.extend(Y_batch.numpy())

    val_acc = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Accuracy: {val_acc:.4f}")

"""##Visualise Training Progress"""

# 4. Plot Loss and Accuracy
plt.plot(train_losses, label='Train Loss')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title("ConvNeXt Training")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.grid(True)
plt.legend()
plt.show()

"""##Evaluate on Test Set"""

# 5. Load Best Model and Evaluate
model.load_state_dict(best_model_state)
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_targets.extend(Y_batch.numpy())

"""##Calculate Test Metrics"""

# 6. Metrics
print("\n--- Final Test Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
print("Accuracy: ", accuracy_score(all_targets, all_preds))
print("Precision: ", precision_score(all_targets, all_preds, average='macro'))
print("Recall: ", recall_score(all_targets, all_preds, average='macro'))
print("F1 Score: ", f1_score(all_targets, all_preds, average='macro'))

all_loss_histories["ConvNeXt"] = train_losses

"""##Save Results"""

# 7. Save results
final_epoch = epochs
final_loss = train_losses[-1]
final_val_acc = val_accuracies[-1]
final_test_acc = accuracy_score(all_targets, all_preds)
final_precision = precision_score(all_targets, all_preds, average='macro')
final_recall = recall_score(all_targets, all_preds, average='macro')
final_f1 = f1_score(all_targets, all_preds, average='macro')

this_model_results = {
    "Model": "ConvNeXt",
    "Epochs": final_epoch,
    "Final Train Loss": round(final_loss, 4),
    "Val Accuracy": round(final_val_acc, 4),
    "Test Accuracy": round(final_test_acc, 4),
    "Precision": round(final_precision, 4),
    "Recall": round(final_recall, 4),
    "F1 Score": round(final_f1, 4)
}

all_model_results.append(this_model_results)

"""#**Results and Visualisation**
This section compares all model performances using charts and also shows correct vs. incorrect predictions using the trained ConvNeXt model.

##Plot: Training Loss for All Models
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
for model_name, losses in all_loss_histories.items():
    plt.plot(losses, label=model_name)
plt.title("Training Loss Curve for All Models")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""##Plot: Metric Comparison in Subplots

"""

import matplotlib.pyplot as plt
import pandas as pd

# Assume df_results is already created from all_model_results
df_results = pd.DataFrame(all_model_results)

# Set plot style
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df_results["Model"], df_results["Test Accuracy"], marker='o', linestyle='-')
plt.title("Test Accuracy")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(df_results["Model"], df_results["Precision"], marker='o', color='green', linestyle='-')
plt.title("Precision")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(df_results["Model"], df_results["Recall"], marker='o', color='red', linestyle='-')
plt.title("Recall")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(df_results["Model"], df_results["F1 Score"], marker='o', color='purple', linestyle='-')
plt.title("F1 Score")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()
plt.show()

"""##Plot: Combined Line Chart for All Metrics"""

import matplotlib.pyplot as plt
import pandas as pd

# Make sure df_results is created from your saved results
df_results = pd.DataFrame(all_model_results)

# === Combined Line Chart ===
plt.figure(figsize=(6, 4))

plt.plot(df_results["Model"], df_results["Test Accuracy"], marker='o', label='Test Accuracy')
plt.plot(df_results["Model"], df_results["Precision"], marker='o', label='Precision')
plt.plot(df_results["Model"], df_results["Recall"], marker='o', label='Recall')
plt.plot(df_results["Model"], df_results["F1 Score"], marker='o', label='F1 Score')

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""##Plot: Combined Line Chart for All Metrics"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your data (replace with your actual DataFrame if needed)
df_results = pd.DataFrame(all_model_results)

# Melt the DataFrame for Seaborn
df_melted = df_results.melt(
    id_vars='Model',
    value_vars=["Test Accuracy", "Precision", "Recall", "F1 Score"],
    var_name='Metric',
    value_name='Score'
)

# Set theme
sns.set_theme(style="whitegrid")

# Define a custom color palette
custom_palette = {
    "Test Accuracy": "#003366",  # yellow
    "Precision":  "#0059b3",     # pink
    "Recall": "#1a8cff",         # cyan
    "F1 Score": "#80bfff"        # blue
}

# Create the bar plot
plt.figure(figsize=(8, 6))
bar_plot = sns.barplot(
    data=df_melted,
    x='Model',
    y='Score',
    hue='Metric',
    palette=custom_palette
)

# Beautify
plt.title("Model Performance Comparison", fontsize=14, fontweight='bold')
plt.ylim(0.6, 1)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Score", fontsize=12)
# plt.xticks(rotation=15)

# Legend inside plot (top-left corner)
plt.legend(title='Metric', loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=True)

plt.tight_layout()
plt.show()

"""##View: All Model Results"""

all_model_results

"""##Plot: Correct vs Incorrect Predictions (ConvNeXt)"""

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

# === De-normalise for visualisation ===
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return tensor * std + mean

# === Use your trained ConvNeXt model ===
model = get_convnext_model(num_classes=5).to(device)
model.load_state_dict(best_model_state)
model.eval()

# === Get 5 correct and 5 incorrect predictions ===
correct = []
incorrect = []

with torch.no_grad():
    for X_batch, Y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1).cpu()
        X_batch = X_batch.cpu()
        Y_batch = Y_batch.cpu()

        for img, true_label, pred_label in zip(X_batch, Y_batch, preds):
            if len(correct) < 5 and pred_label == true_label:
                correct.append((img, pred_label.item(), true_label.item()))
            elif len(incorrect) < 5 and pred_label != true_label:
                incorrect.append((img, pred_label.item(), true_label.item()))
            if len(correct) >= 5 and len(incorrect) >= 5:
                break
        if len(correct) >= 5 and len(incorrect) >= 5:
            break

# === Plot the images ===
fig, axs = plt.subplots(2, 5, figsize=(7, 4))  # smaller grid

for i in range(5):
    # Correct prediction
    img, pred, true = correct[i]
    img = denormalize(img)
    axs[0, i].imshow(to_pil_image(img))
    axs[0, i].set_title(f"{selected_classes[pred]}", fontsize=10, color='green')
    axs[0, i].axis('off')

    # Incorrect prediction
    img, pred, true = incorrect[i]
    img = denormalize(img)
    axs[1, i].imshow(to_pil_image(img))
    axs[1, i].set_title(f"{selected_classes[pred]} (was {selected_classes[true]})", fontsize=9, color='red')
    axs[1, i].axis('off')

plt.suptitle("Top: Correct Predictions   |   Bottom: Incorrect Predictions", fontsize=12)
plt.tight_layout()
plt.show()
