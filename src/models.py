"""
One configurable encoder-decoder used for all four models.

    skip_mode : "none"   -> plain encoder-decoder (baseline, no skip connections)
                "concat" -> U-Net style: decoder input = [upsampled, encoder feature]
                "add"    -> decoder input = upsampled + encoder feature (element-wise)
    up_mode   : "interp" -> bilinear x2 interpolation followed by a 1x1 conv
                "tconv"  -> 2x2 transposed convolution with stride 2

Everything else (depth, channel widths, conv blocks, BatchNorm, head) is
identical, so any difference in the scores comes only from the skip
connection / upsampling choice.

Channel widths with base=32, depth=4 and a 256x256 input:

    enc1  32 @256  ──────────────────────────── skip ──┐
    enc2  64 @128  ─────────────────────── skip ──┐    │
    enc3 128 @64   ────────────────── skip ──┐    │    │
    enc4 256 @32   ───────────── skip ──┐    │    │    │
    bottleneck 512 @16                  ▼    ▼    ▼    ▼
                               dec4 256@32 dec3 128@64 dec2 64@128 dec1 32@256 -> 1x1 conv -> logit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2 — the standard U-Net block."""

    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """x2 upsampling that also halves the number of channels."""

    def __init__(self, cin, cout, mode):
        super().__init__()
        self.mode = mode
        if mode == "tconv":
            # learnable upsampling
            self.op = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        elif mode == "interp":
            # fixed bilinear upsampling, then a 1x1 conv to change channel count
            self.op = nn.Conv2d(cin, cout, kernel_size=1)
        else:
            raise ValueError(mode)

    def forward(self, x):
        if self.mode == "interp":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.op(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, skip_mode, up_mode):
        super().__init__()
        self.skip_mode = skip_mode
        self.up = Up(cin, cout, up_mode)
        # concatenation doubles the channels going into the conv block;
        # "add" and "none" keep them the same (so P2 has exactly the baseline's params)
        conv_in = 2 * cout if skip_mode == "concat" else cout
        self.conv = DoubleConv(conv_in, cout)

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.skip_mode == "concat":
            x = torch.cat([x, skip], dim=1)
        elif self.skip_mode == "add":
            x = x + skip
        return self.conv(x)


class EncoderDecoder(nn.Module):
    def __init__(self, skip_mode="concat", up_mode="interp", in_ch=3, base=32, depth=4):
        super().__init__()
        assert skip_mode in ("none", "concat", "add")
        self.skip_mode, self.up_mode, self.depth = skip_mode, up_mode, depth
        chs = [base * 2 ** i for i in range(depth + 1)]          # e.g. 32,64,128,256,512

        self.enc = nn.ModuleList()
        cin = in_ch
        for c in chs[:-1]:
            self.enc.append(DoubleConv(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        self.dec = nn.ModuleList(
            DecoderBlock(chs[i + 1], chs[i], skip_mode, up_mode) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chs[0], 1, kernel_size=1)          # 1 logit per pixel (horse vs bg)

    def forward(self, x, drop_skips=()):
        """
        drop_skips: decoder levels whose skip feature is replaced by zeros.
        Level 1 = highest resolution (256x256), level `depth` = lowest.
        Only used for the test-time skip knock-out analysis; training never sets it.
        """
        feats = []
        for enc in self.enc:
            x = enc(x)
            feats.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        for i, dec in enumerate(self.dec):
            level = self.depth - i                                # 4,3,2,1
            skip = None
            if self.skip_mode != "none":
                skip = feats[level - 1]
                if level in drop_skips:
                    skip = torch.zeros_like(skip)
            x = dec(x, skip)
        return self.head(x)


def build_model(model_key, cfg):
    from .utils import MODELS
    _, skip_mode, up_mode = MODELS[model_key]
    m = cfg["model"]
    return EncoderDecoder(skip_mode=skip_mode, up_mode=up_mode,
                          base=m.get("base_channels", 32), depth=m.get("depth", 4))
