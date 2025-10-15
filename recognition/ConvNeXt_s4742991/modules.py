import torch
import torch.nn as nn

# DropPath (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask

# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path)

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
        return residual + self.drop_path(x)

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
    def __init__(self, in_chans=3, num_classes=1, depths=[2, 2, 3, 2], dims=[32, 64, 128, 256], drop_path_rate=0.1):
        super().__init__()
        self.stem = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.stem_norm = nn.LayerNorm(dims[0], eps=1e-6)

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(DownsampleLayer(dims[i], dims[i+1]))

        # Compute per-block drop path rates
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur]))
                cur += 1
            self.stages.append(nn.Sequential(*blocks))

        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=0.3)
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
        x = self.dropout(x)
        x = self.head(x)
        return x.squeeze(1)

# Model factory
def get_model():
    return ConvNeXt()

# Parameter counter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
