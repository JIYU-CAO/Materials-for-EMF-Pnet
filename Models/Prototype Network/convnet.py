import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
        )

        self.embedding = nn.Linear(1600, z_dim)
        self.out_channels = 1600  # 展平后的特征维度

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
