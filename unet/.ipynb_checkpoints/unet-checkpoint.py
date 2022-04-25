import torch
import torch.nn as nn
import torchvision
from netCDF4 import Dataset
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import tempfile
import sys
import torch.nn.functional as F


class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None, first_kernel=3, second_kernel=3, padding=0):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.first_down = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, first_kernel, padding=padding),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, second_kernel, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.first_down(x)
    
    
class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)
    
    
class Up(nn.Module):
    # Taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,padding=1)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,padding=1)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # n,c,z,y,x
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    
class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 32, first_kernel=7, padding=2)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.outc = OutConv(32, 2)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits
    
    def test_correct_size(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        
        return x