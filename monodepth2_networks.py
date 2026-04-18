from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if use_refl else nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        return self.nonlin(self.conv(x))


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=False):
        super().__init__()
        resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}
        self.encoder = resnets[num_layers](weights=None if not pretrained else "DEFAULT")
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        feats = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        feats.append(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x); feats.append(x)
        x = self.encoder.layer2(x); feats.append(x)
        x = self.encoder.layer3(x); feats.append(x)
        x = self.encoder.layer4(x); feats.append(x)
        return feats


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = list(scales)
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = int(self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1])
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, int(self.num_ch_dec[i]))

            num_ch_in = int(self.num_ch_dec[i])
            if self.use_skips and i > 0:
                num_ch_in += int(self.num_ch_enc[i - 1])
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, int(self.num_ch_dec[i]))

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(int(self.num_ch_dec[s]), 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [nn.functional.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
        return outputs
