import torch.nn as nn
import torch.functional as F
from typing import Any
import torchview
from torch import Tensor
import torch


class Resnet(nn.Module):
    def __init__(
        self, num_groups, in_channels, out_channels, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        down_sample: bool,
        num_heads: int,
        num_layers: int,
        attn: bool,
        norm_channels: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.attn = attn
        self.down_sample = down_sample

        self.resnet_conv_first = nn.ModuleList(
            [
                Resnet(
                    norm_channels,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(self.t_emb_dim, out_channels))
                    for _ in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                Resnet(norm_channels, out_channels, out_channels)
                for i in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for x in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for x in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if x == 0 else out_channels, out_channels, kernel_size=1
                )
                for x in range(num_layers)
            ]
        )

        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                # Attention Block
                batch_size, channels, h, w = out.shape
                in_attn: Tensor = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                # print(type(out_attn), type(ssss))
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return self.down_sample_conv(out)


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        t_emb_dim: int,
        num_heads: int,
        num_layers: int,
        norm_channels: int,
        attn: bool = True,
        cross_attn: bool = False,
        context_dim: int = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.attn = attn
        self.cross_attn = cross_attn

        self.resnet_conv_first = nn.ModuleList(
            [
                Resnet(
                    norm_channels,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                )
                for i in range(num_layers + 1)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(self.t_emb_dim, out_channels))
                    for _ in range(num_layers + 1)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                Resnet(norm_channels, out_channels, out_channels)
                for i in range(num_layers + 1)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for x in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for x in range(num_layers)
                ]
            )

        if self.cross_attn:
            assert context_dim is not None, (
                "Context Dimension must be passed for cross attention"
            )
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels) for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if x == 0 else out_channels, out_channels, kernel_size=1
                )
                for x in range(num_layers + 1)
            ]
        )

    def forward(self, x, t_emb=None, context=None):
        out = x
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            batch_size, channels, h, w = out.shape
            in_attn: Tensor = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            # print(type(out_attn), type(ssss))
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None, (
                    "context cannot be None if cross attention layers are used"
                )
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                )
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            # following resnet blocks
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up_sample: bool,
        num_heads: int,
        num_layers: int,
        attn: bool,
        norm_channels: int,
        t_emb_dim: int = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.num_heads = num_heads
        self.attn = attn
        self.up_sample = up_sample

        self.resnet_conv_first = nn.ModuleList(
            [
                Resnet(
                    norm_channels,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                )
                for i in range(num_layers)
            ]
        )
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(self.t_emb_dim, out_channels))
                    for _ in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                Resnet(norm_channels, out_channels, out_channels)
                for i in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for x in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for x in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if x == 0 else out_channels, out_channels, kernel_size=1
                )
                for x in range(num_layers)
            ]
        )

        self.up_sample_conv = (
            nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=4, stride=2, padding=1
            )
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x, out_down=None, t_emb=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                # Attention Block
                batch_size, channels, h, w = out.shape
                in_attn: Tensor = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                # print(type(out_attn), type(ssss))
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
        return out


class UpBlockUnet(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        up_sample,
        num_heads,
        num_layers,
        norm_channels,
        cross_attn=False,
        context_dim=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                    for _ in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        if self.cross_attn:
            assert context_dim is not None, (
                "Context Dimension must be passed for cross attention"
            )
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels) for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = (
            nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)
            if self.up_sample
            else nn.Identity()
        )

    def forward(self, x, out_down=None, t_emb=None, context=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self Attention
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            # Cross Attention
            if self.cross_attn:
                assert context is not None, (
                    "context cannot be None if cross attention layers are used"
                )
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert len(context.shape) == 3, (
                    "Context shape does not match B,_,CONTEXT_DIM"
                )
                assert (
                    context.shape[0] == x.shape[0]
                    and context.shape[-1] == self.context_dim
                ), "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](
                    in_attn, context_proj, context_proj
                )
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out
