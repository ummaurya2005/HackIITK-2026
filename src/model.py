import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, input_dim=120, num_classes=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),      # net.0
            nn.BatchNorm1d(256),            # net.1
            nn.ReLU(),                      # net.2
            nn.Dropout(0.4),                # net.3

            nn.Linear(256, 128),             # net.4
            nn.BatchNorm1d(128),             # net.5
            nn.ReLU(),                      # net.6
            nn.Dropout(0.3),                # net.7

            nn.Linear(128, num_classes)     # net.8
        )

    def forward(self, x):
        return self.net(x)
