import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, input_features, expansion=0.25):
        super().__init__()
        self.hidden_size = max(1, int(input_features * expansion))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.fc = nn.Sequential(
            nn.Linear(input_features, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, input_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale: torch.Tensor = self.fc(self.pool(x))
        return x * scale[:, :, None, None]


class PreNormalization(nn.Module):
    def __init__(self, num_features, module: nn.Module, norm: nn.Module):
        super().__init__()
        self.norm = norm(num_features)
        self.module = module

    def forward(self, x):
        return self.module(self.norm(x))


class MBConv(nn.Module):
    def __init__(self, input_features, output_features, expansion_rate=4,
                 shrinkage_rate=0.25, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.hidden_dim = int(input_features * expansion_rate)
        self.stride = 2 if downsample else 1
        self.project = input_features != output_features

        if downsample:
            self.identity_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.identity_pool = nn.Identity()

        if self.project or downsample:
            self.identity_proj = nn.Conv2d(input_features, output_features,
                                           kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.identity_proj = nn.Identity()

        self.conv = PreNormalization(input_features, nn.Sequential(
            nn.Conv2d(input_features, self.hidden_dim, kernel_size=1,
                      stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3,
                      stride=1, padding=1, groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            SqueezeExcitation(self.hidden_dim, expansion=shrinkage_rate),
            nn.Conv2d(self.hidden_dim, output_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_features),
        ), nn.BatchNorm2d)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.identity_proj(self.identity_pool(x))
        elif self.project:
            identity = self.identity_proj(x)
        return identity + self.conv(x)


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, feat_size, input_features):
        super().__init__()
        self.d_model = input_features
        self.num_heads = self.d_model // 32
        self.d_k = 32

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=True)

        self.feat_size = feat_size
        num_rel = (2 * feat_size - 1) ** 2
        self.relative_bias_table = nn.Parameter(torch.zeros(self.num_heads, num_rel))
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

        coords_h = torch.arange(feat_size)
        coords_w = torch.arange(feat_size)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
        coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)

        rel = coords[:, None, :] - coords[None, :, :]
        rel[:, :, 0] += feat_size - 1
        rel[:, :, 1] += feat_size - 1
        rel_index = rel[:, :, 0] * (2 * feat_size - 1) + rel[:, :, 1]

        self.register_buffer("relative_position_index", rel_index.long())

    def forward(self, x):
        B, N, C = x.shape

        q = self.W_q(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).reshape(B, N, self.num_heads, self.d_k).transpose(1, 2)

        rel_bias = self.relative_bias_table[:, self.relative_position_index.view(-1)]
        rel_bias = rel_bias.view(1, self.num_heads, N, N)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_bias)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.W_o(out)


class TransformerDownsampleBlock(nn.Module):
    def __init__(self, feat_size, input_features, output_features, expansion_rate=4):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.identity_proj = nn.Conv2d(input_features, output_features,
                                       kernel_size=1, bias=False)

        self.attn_norm = nn.LayerNorm(input_features)
        self.attn = RelativeMultiHeadAttention(feat_size, input_features)
        self.attn_proj = nn.Linear(input_features, output_features)

        self.ffn_norm = nn.LayerNorm(output_features)
        self.ffn = nn.Sequential(
            nn.Linear(output_features, int(output_features * expansion_rate)),
            nn.GELU(),
            nn.Linear(int(output_features * expansion_rate), output_features),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        identity = self.identity_proj(self.pool(x))
        identity = identity.flatten(2).transpose(1, 2)

        res = x.flatten(2).transpose(1, 2)
        res = self.attn_norm(res)
        res = res.transpose(1, 2).reshape(B, C, H, W)
        res = self.pool(res).flatten(2).transpose(1, 2)
        res = self.attn_proj(self.attn(res))

        x = identity + res
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, feat_size, dim, expansion_rate=4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(feat_size, dim)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * expansion_rate)),
            nn.GELU(),
            nn.Linear(int(dim * expansion_rate), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CoAtNet0(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, image_size=224):
        super().__init__()

        dims = [64, 96, 192, 384, 768]
        depths = [2, 2, 3, 5, 2]

        s3_size = image_size // 16
        s4_size = image_size // 32

        self.s0 = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        self.s1 = nn.Sequential(
            MBConv(dims[0], dims[1], downsample=True),
            *[MBConv(dims[1], dims[1]) for _ in range(depths[1] - 1)],
        )

        self.s2 = nn.Sequential(
            MBConv(dims[1], dims[2], downsample=True),
            *[MBConv(dims[2], dims[2]) for _ in range(depths[2] - 1)],
        )

        self.s3_down = TransformerDownsampleBlock(s3_size, dims[2], dims[3])
        self.s3 = nn.Sequential(
            *[TransformerBlock(s3_size, dims[3]) for _ in range(depths[3] - 1)],
        )

        self.s4_down = TransformerDownsampleBlock(s4_size, dims[3], dims[4])
        self.s4 = nn.Sequential(
            *[TransformerBlock(s4_size, dims[4]) for _ in range(depths[4] - 1)],
        )

        self.s3_size = s3_size
        self.head = nn.Sequential(
            nn.LayerNorm(dims[4]),
            nn.Linear(dims[4], num_classes),
        )

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)

        x = self.s3_down(x)
        x = self.s3(x)

        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.s3_size, self.s3_size)
        x = self.s4_down(x)
        x = self.s4(x)

        x = x.mean(dim=1)
        return self.head(x)
