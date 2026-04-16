import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM_Model(nn.Module):
    def __init__(self, hidden_size=128, num_classes=3):
        super(CNN_LSTM_Model, self).__init__()

        # Pretrained CNN (feature extractor)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove classification layer

        # LSTM for temporal learning
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size, 
            num_layers=2, 
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, sequence, channels, height, width)

        batch_size, seq_len, C, H, W = x.size()

        cnn_out = []
        for t in range(seq_len):
            out = self.cnn(x[:, t, :, :, :])  # (batch, 512)
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)  # (batch, seq, 512)

        lstm_out, _ = self.lstm(cnn_out)
        final_out = lstm_out[:, -1, :]  # Last time step

        return self.fc(final_out)
      
