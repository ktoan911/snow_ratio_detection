"""
U-Net architecture theo paper:
- Encoder (contracting path): 2×(Conv3x3+BN+ReLU) → MaxPool2d
  Channels: 3 → 64 → 128 → 256 → 512 → bottleneck 1024
- Decoder (expansive path): bilinear upsample + skip-concat + 2×(Conv3x3+BN+ReLU)
- Output: 1 channel (binary segmentation) → dùng với BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Khối conv cơ bản: 2×(Conv3x3 → BN → ReLU) ──────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Encoder block: DoubleConv → MaxPool ─────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)  # feature map giữ lại cho skip connection
        down = self.pool(skip)
        return skip, down


# ── Decoder block: bilinear upsample → concat skip → DoubleConv ─────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch = channels từ upsample + channels skip (bằng nhau → in_ch = 2×out_ch)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        # Bilinear upsample ×2
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)  # skip connection (concat)
        return self.conv(x)


# ── U-Net chính ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net chuẩn 4 lần downsample.
    Channel progression:
        Encoder: 3 → 64 → 128 → 256 → 512
        Bottleneck: 512 → 1024
        Decoder: 1024+512→512 → 512+256→256 → 256+128→128 → 128+64→64
        Output: 64 → 1 (binary mask, raw logits)
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, base_features: int = 64
    ):
        super().__init__()

        f = base_features  # 64

        # Encoder
        self.enc1 = EncoderBlock(in_channels, f)  # 64
        self.enc2 = EncoderBlock(f, f * 2)  # 128
        self.enc3 = EncoderBlock(f * 2, f * 4)  # 256
        self.enc4 = EncoderBlock(f * 4, f * 8)  # 512

        # Bottleneck (không pool)
        self.bottleneck = DoubleConv(f * 8, f * 16)  # 1024

        # Decoder
        # Mỗi DecoderBlock nhận (upsample_ch + skip_ch, out_ch)
        self.dec4 = DecoderBlock(f * 16 + f * 8, f * 8)  # 1024+512 → 512
        self.dec3 = DecoderBlock(f * 8 + f * 4, f * 4)  # 512+256  → 256
        self.dec2 = DecoderBlock(f * 4 + f * 2, f * 2)  # 256+128  → 128
        self.dec1 = DecoderBlock(f * 2 + f, f)  # 128+64   → 64

        # Output head
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Output (raw logits — dùng BCEWithLogitsLoss)
        return self.out_conv(x)


# # ── Quick sanity check ───────────────────────────────────────────────────────
# if __name__ == "__main__":
#     model = UNet(in_channels=3, out_channels=1)
#     dummy = torch.randn(2, 3, 512, 512)
#     out = model(dummy)
#     print(f"Input : {dummy.shape}")
#     print(f"Output: {out.shape}")  # expect (2, 1, 512, 512)
#     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable params: {params:,}")
