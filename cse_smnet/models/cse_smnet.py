# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:03:30 2026

@author: USER
"""

import torch
import torch.nn as nn

from .encoder import EncoderStage
from .decoder import DecoderStage
from .temporal.timestamp import TimestampTransform
from .temporal.mtar import MTAR


###############################################
# Full Autoencoder Model with Skip Connections
###############################################
class CSESMNet(nn.Module):
    """CSE-SMNet future-frame prediction network.

    Registered layer names are intentionally kept compatible with the trained
    research checkpoint; public module/class terminology follows the revised paper.
    """

    def __init__(self, seq_len=3):

        super().__init__()

        self.seq_len = seq_len

        ###############################################
        # Initial Feature Extraction
        ###############################################

        self.initial = nn.Conv2d(
            3,
            24,
            3,
            padding=1
        )

        ###############################################
        # Encoder
        ###############################################

        self.enc1 = EncoderStage(24, 48)
        self.enc2 = EncoderStage(48, 96)
        self.enc3 = EncoderStage(96, 192)
        self.enc4 = EncoderStage(192, 384)
       # self.enc4 = EncoderStage(192, 256)

        ###############################################
        # Timestamp Transformation
        ###############################################

        self.timestamp = TimestampTransform()

        ###############################################
        # Temporal Modeling
        ###############################################

        self.e3d = MTAR(384, 384)
        # self.e3d = MTAR(256, 256)

        ###############################################
        # Decoder
        ###############################################

        self.dec1 = DecoderStage(384, 192)
        #self.dec1 = DecoderStage(256, 192)
        self.dec2 = DecoderStage(192, 96)
        self.dec3 = DecoderStage(96, 48)
        self.dec4 = DecoderStage(48, 24)

        ###############################################
        # Skip Fusion Layers
        ###############################################

        self.skip1 = nn.Conv2d(192 + 192, 192, 1)
        self.skip2 = nn.Conv2d(96 + 96, 96, 1)
        self.skip3 = nn.Conv2d(48 + 48, 48, 1)
        self.skip4 = nn.Conv2d(24 + 24, 24, 1)

        ###############################################
        # Final Reconstruction
        ###############################################

        self.final = nn.Conv2d(
            24,
            3,
            3,
            padding=1
        )

    @property
    def mtar(self):
        """Public MTAR name; legacy `e3d` key is retained for checkpoint compatibility."""
        return self.e3d

    def forward(self, x):

        B, T, C, H, W = x.shape

        ###############################################
        # Merge Batch and Time
        ###############################################

        x = x.view(B * T, C, H, W)

        ###############################################
        # Initial Feature Extraction
        ###############################################

        x0 = self.initial(x)

        ###############################################
        # Encoder
        ###############################################

        e1 = self.enc1(x0)   # [BT,48,128,128]
        e2 = self.enc2(e1)   # [BT,96,64,64]
        e3 = self.enc3(e2)   # [BT,192,32,32]
        e4 = self.enc4(e3)   # [BT,384,16,16]

        ###############################################
        # Convert to Spatiotemporal Tensor
        # (B,C,T,H,W)
        ###############################################

        x = self.timestamp(e4, B, T)

        ###############################################
        # Temporal Modeling
        ###############################################

        x = self.mtar(x)

        ###############################################
        # Take Last Time Step
        ###############################################

        x = x[:, :, -1]   # [B,384,16,16]

        ###############################################
        # Last Encoder Features
        ###############################################

        e3_last = (
            e3.view(
                B,
                T,
                192,
                e3.shape[-2],
                e3.shape[-1]
            )[:, -1]
        )

        e2_last = (
            e2.view(
                B,
                T,
                96,
                e2.shape[-2],
                e2.shape[-1]
            )[:, -1]
        )

        e1_last = (
            e1.view(
                B,
                T,
                48,
                e1.shape[-2],
                e1.shape[-1]
            )[:, -1]
        )

        x0_last = (
            x0.view(
                B,
                T,
                24,
                x0.shape[-2],
                x0.shape[-1]
            )[:, -1]
        )

        ###############################################
        # Decoder Stage 1
        ###############################################

        x = self.dec1(x)              # [B,192,32,32]

        x = torch.cat(
            [x, e3_last],
            dim=1
        )

        x = self.skip1(x)

        ###############################################
        # Decoder Stage 2
        ###############################################

        x = self.dec2(x)              # [B,96,64,64]

        x = torch.cat(
            [x, e2_last],
            dim=1
        )

        x = self.skip2(x)

        ###############################################
        # Decoder Stage 3
        ###############################################

        x = self.dec3(x)              # [B,48,128,128]

        x = torch.cat(
            [x, e1_last],
            dim=1
        )

        x = self.skip3(x)

        ###############################################
        # Decoder Stage 4
        ###############################################

        x = self.dec4(x)              # [B,24,256,256]

        x = torch.cat(
            [x, x0_last],
            dim=1
        )

        x = self.skip4(x)

        ###############################################
        # Final Reconstruction
        ###############################################

        x = self.final(x)

        return x
