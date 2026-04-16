from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from hybrid_model import CNN_LSTM_Model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
BATCH_SIZE = 4
SEQ_LEN = 5
NUM_CLASSES = 3
IMG_SIZE = 224

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Load Dataset
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

# Prepare data
sequences, labels = create_sequences(dataset, SEQ_LEN)

X = torch.stack(sequences)
y = torch.tensor(labels)

test_loader = DataLoader(list(zip(X, y)), batch_size=BATCH_SIZE)

# Load model
model = CNN_LSTM_Model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("models/saved_models/cnn_lstm.pth"))
model.eval()

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
