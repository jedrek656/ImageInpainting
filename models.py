import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.ConvBlock(4, 32)
        self.enc2 = self.ConvBlock(32, 64)
        self.enc3 = self.ConvBlock(64, 128)
        self.enc4 = self.ConvBlock(128, 256)

        self.bottleNeck = self.BottleNeckBlock(256, 512)

        self.up4 = self.upBlock(256)
        self.dec4 = self.ConvBlock(256 * 2, 128)

        self.up3 = self.upBlock(128)
        self.dec3 = self.ConvBlock(128*2, 64)

        self.up2 = self.upBlock(64)
        self.dec2 = self.ConvBlock(64*2, 32)

        self.up1 = self.upBlock(32)
        self.dec1 = self.ConvBlock(32*2, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def BottleNeckBlock(self, in_channels, inner_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(inner_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

    def ConvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    
    def upBlock(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleNeck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim = 1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim = 1)
        d2 = self.dec3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim = 1)
        d1 = self.dec2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim = 1)
        d0 = self.dec1(d1)

        return self.final(d0)

class simpleAEsmaller(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.ConvBlock(4, 32)
        self.enc2 = self.ConvBlock(32, 64)
        self.enc3 = self.ConvBlock(64, 128)

        self.bottleNeck = self.BottleNeckBlock(128, 256)

        self.up3 = self.upBlock(128)
        self.dec3 = self.ConvBlock(128*2, 64)

        self.up2 = self.upBlock(64)
        self.dec2 = self.ConvBlock(64*2, 32)

        self.up1 = self.upBlock(32)
        self.dec1 = self.ConvBlock(32*2, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def BottleNeckBlock(self, in_channels, inner_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(inner_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

    def ConvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    
    def upBlock(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleNeck(F.max_pool2d(e3, 2))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim = 1)
        d2 = self.dec3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim = 1)
        d1 = self.dec2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim = 1)
        d0 = self.dec1(d1)

        return self.final(d0)
    
class simpleAEevenSmaller(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.ConvBlock(4, 32)
        self.enc2 = self.ConvBlock(32, 64)
        self.enc3 = self.ConvBlock(64, 96)

        self.bottleNeck = self.BottleNeckBlock(96)

        self.up3 = self.upBlock(96)
        self.dec3 = self.ConvBlock(96*2, 64)

        self.up2 = self.upBlock(64)
        self.dec2 = self.ConvBlock(64*2, 32)

        self.up1 = self.upBlock(32)
        self.dec1 = self.ConvBlock(32*2, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def BottleNeckBlock(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

    def ConvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    
    def upBlock(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleNeck(F.max_pool2d(e3, 2))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim = 1)
        d2 = self.dec3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim = 1)
        d1 = self.dec2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim = 1)
        d0 = self.dec1(d1)

        return self.final(d0)

class simpleAEevenSmallerWithUpsampling(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.ConvBlock(4, 32)
        self.enc2 = self.ConvBlock(32, 64)
        self.enc3 = self.ConvBlock(64, 96)

        self.bottleNeck = self.BottleNeckBlock(96)

        self.up3 = self.upBlock(96)
        self.dec3 = self.ConvBlock(96*2, 64)

        self.up2 = self.upBlock(64)
        self.dec2 = self.ConvBlock(64*2, 32)

        self.up1 = self.upBlock(32)
        self.dec1 = self.ConvBlock(32*2, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def BottleNeckBlock(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=4, dilation=4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=8, dilation=8),
            nn.LeakyReLU(0.1),
        )

    def ConvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )
    
    def upBlock(self, in_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleNeck(F.max_pool2d(e3, 2))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim = 1)
        d2 = self.dec3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim = 1)
        d1 = self.dec2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim = 1)
        d0 = self.dec1(d1)

        return self.final(d0)
    
class noBatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = self.ConvBlock(4, 32)
        self.enc2 = self.ConvBlock(32, 64)
        self.enc3 = self.ConvBlock(64, 96)

        self.bottleNeck = self.BottleNeckBlock(96)

        self.up3 = self.upBlock(96)
        self.dec3 = self.ConvBlock(96*2, 64)

        self.up2 = self.upBlock(64)
        self.dec2 = self.ConvBlock(64*2, 32)

        self.up1 = self.upBlock(32)
        self.dec1 = self.ConvBlock(32*2, 32)

        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def BottleNeckBlock(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=4, dilation=4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=8, dilation=8),
            nn.LeakyReLU(0.1),
        )

    def ConvBlock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
    
    def upBlock(self, in_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleNeck(F.max_pool2d(e3, 2))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim = 1)
        d2 = self.dec3(d3)

        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim = 1)
        d1 = self.dec2(d2)

        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim = 1)
        d0 = self.dec1(d1)

        return self.final(d0)