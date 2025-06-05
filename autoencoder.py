from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class StrongUNetNoSkips(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck with residuals
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(1024),
        )

        # Decoder (no skips)
        self.up4 = self.up_block(1024, 512)
        self.dec4 = self.conv_block(512, 512)

        self.up3 = self.up_block(512, 256)
        self.dec3 = self.conv_block(256, 256)

        self.up2 = self.up_block(256, 128)
        self.dec2 = self.conv_block(128, 128)

        self.up1 = self.up_block(128, 64)
        self.dec1 = self.conv_block(64, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)

        x = self.enc2(x)
        x = self.pool(x)

        x = self.enc3(x)
        x = self.pool(x)

        x = self.enc4(x)
        x = self.pool(x)

        x = self.bottleneck(x)

        x = self.up4(x)
        x = self.dec4(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.final_conv(x)