import torch
import torch.nn.functional as F
import torch.nn as nn
from param import args


class Discriminator(nn.Module):
    def __init__(self, dim=args.feature_size, classes=62):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.classes = classes
        self.classifier = nn.Sequential(
            nn.Linear(dim, 768),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(768, 768),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(768, 61),     # 61 scans in training set
            nn.Softmax()
        )

    def forward(self, input):
        b, n, d = input.shape
        input = input.reshape(-1, d)

        probs = self.classifier(input)

        # probs = probs.reshape(b, n, self.classes)
        return probs

