"""Two small models that take a log-mel input of shape (B, 1, n_mels, T).

SmallCNN : 4 conv blocks + global average pooling. Fully convolutional,
           so it works for any clip length.
CRNN     : 2 conv blocks + bidirectional GRU over time. Usually better at
           temporal patterns, but recurrent layers can behave differently
           under INT8 quantization - that is worth measuring, not assuming.
"""
import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), conv_block(64, 128)
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.features(x)            # (B, 128, F', T')
        x = x.mean(dim=(2, 3))          # global average pooling
        return self.head(x)


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 8, n_mels: int = 64, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(conv_block(1, 16), conv_block(16, 32))
        self.gru = nn.GRU(32 * (n_mels // 4), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x):
        x = self.conv(x)                          # (B, 32, F/4, T/4)
        x = x.permute(0, 3, 1, 2).flatten(2)      # (B, T/4, 32 * F/4)
        x, _ = self.gru(x)
        return self.head(x.mean(dim=1))


def build_model(name: str, n_classes: int = 8, n_mels: int = 64) -> nn.Module:
    if name == "cnn":
        return SmallCNN(n_classes)
    if name == "crnn":
        return CRNN(n_classes, n_mels)
    raise ValueError(f"unknown model '{name}' (choose: cnn, crnn)")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
