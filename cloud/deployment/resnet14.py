import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
  layers = [
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, ),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
  ]
  if pool: layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super(ResidualBlock, self).__init__()
    self.conv = nn.Sequential(
      conv_block(channels, channels),
      conv_block(channels, channels)
    )

  def forward(self, x):
    return self.conv(x) + x


class ResNet14(nn.Module):
  def __init__(self, in_channels, num_classes):
    super().__init__()

    self.conv = nn.Sequential(
        conv_block(in_channels, 64),
        conv_block(64, 128, pool=True),
        conv_block(128, 128, pool=True),
        ResidualBlock(128),

        conv_block(128, 256),
        ResidualBlock(256),
        conv_block(256, 512),
        ResidualBlock(512),
        conv_block(512, 256),
        ResidualBlock(256),
        conv_block(256, 128),
        ResidualBlock(128),

        conv_block(128, 256, pool=True),
        conv_block(256, 512, pool=True),
        ResidualBlock(512),

        nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    )

  def forward(self, xb):
    return self.conv(xb)
