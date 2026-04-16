import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from hybrid_model import CNN_LSTM_Model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
BATCH_SIZE = 4
SEQ_LEN = 5
EPOCHS = 5
NUM_CLASSES = 3
IMG_SIZE = 224

# Image Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load Dataset (assumes folder structure: spectrograms/class_name/images)
dataset = datasets.ImageFolder("data/spectrograms", transform=transform)

def create_sequences(dataset, seq_len):
    sequences = []
    labels = []

    for i in range(len(dataset) - seq_len):
        seq = []
        for j in range(seq_len):
            img, label = dataset[i + j]
            seq.append(img)
        sequences.append(torch.stack(seq))
        labels.append(label)

    return sequences, labels

# Prepare sequences
sequences, labels = create_sequences(dataset, SEQ_LEN)

# Convert to tensors
X = torch.stack(sequences)
y = torch.tensor(labels)

# DataLoader
train_loader = DataLoader(list(zip(X, y)), batch_size=BATCH_SIZE, shuffle=True)

# Model
model = CNN_LSTM_Model(num_classes=NUM_CLASSES).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# Save model
os.makedirs("models/saved_models", exist_ok=True)
torch.save(model.state_dict(), "models/saved_models/cnn_lstm.pth")

print("Model training complete and saved!")
