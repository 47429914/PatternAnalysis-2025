import torch
import torch.nn as nn

# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return residual + x

# Downsampling layer
class DownsampleLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x

# Full ConvNeXt model
class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]):
        super().__init__()
        self.stem = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(dims[0], eps=1e-6)

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(DownsampleLayer(dims[i], dims[i+1]))

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)

        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i](x)

        x = x.mean([-2, -1])  # global average pooling
        x = self.final_norm(x)
        x = self.head(x)
        return x.squeeze(1)

# Model factory
def get_model():
    return ConvNeXt()

# Parameter counter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
