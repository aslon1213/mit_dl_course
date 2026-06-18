from torch import nn
from typing import Any
from blocks.vqvae_blocks import DownBlock, UpBlock, MidBlock
import torch
from torch import Tensor


class VQVAE(nn.Module):
    def __init__(self, im_channels=3, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.down_channels = [64, 128, 256, 256]
        self.mid_channels = [256, 256]
        self.down_sample = [True, True, True]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.norm_channels = 32

        # To disable attention in DownBlock of Encoder and Upblock of Decoder
        self.attns = [False, False, False]
        self.num_heads = 4

        # Latent Dimensions
        self.z_channels = 3
        self.codebook_size = 8192

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ######## Encoder ##########
        self.encoder_conv_in = nn.Conv2d(
            im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1)
        )

        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    t_emb_dim=None,
                    norm_channels=self.norm_channels,
                    down_sample=self.down_sample[i],
                )
            )

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(
            self.down_channels[-1], self.z_channels, kernel_size=3, padding=1
        )

        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

        ######## Decoder ########
        self.post_quant_conv = nn.Conv2d(
            self.z_channels, self.z_channels, kernel_size=1
        )
        # IMPORTANT: Fix decoder_conv_in to match the expected input channels.
        # Before:
        # self.decoder_conv_in = nn.Conv2d(
        #     self.z_channels, self.mid_channels[-1], kernel_size=1, padding=(1, 1)
        # )
        # After: Should map z_channels -> mid_channels[0] not mid_channels[-1]!
        self.decoder_conv_in = nn.Conv2d(
            self.z_channels, self.mid_channels[0], kernel_size=3, padding=1
        )

        self.decoder_mids = nn.ModuleList()
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i - 1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    num_layers=self.num_mid_layers,
                    norm_channels=self.norm_channels,
                )
            )

        self.decoder_ups = nn.ModuleList()
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_ups.append(
                UpBlock(
                    self.down_channels[i],
                    self.down_channels[i - 1],
                    num_heads=self.num_heads,
                    num_layers=self.num_up_layers,
                    attn=self.attns[i - 1],
                    norm_channels=self.norm_channels,
                    up_sample=self.up_sample[i - 1],
                )
            )

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(
            self.down_channels[0],
            im_channels,
            kernel_size=3,
            padding=1,
        )

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H * W, C
        x = x.reshape(B, H * W, C)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((B, 1, 1)))

        min_encoding_indices = torch.argmin(dist, dim=-1)

        quant_out = torch.index_select(
            self.embedding.weight, 0, min_encoding_indices.view(-1)
        )

        x_ = x.reshape((-1, x.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - x_) ** 2)
        codebook_loss = torch.mean((quant_out - x_.detach()) ** 2)

        quantize_losses = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

        quant_out = x_ + (quant_out - x_).detach()
        quant_out = quant_out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return quant_out, quantize_losses

    def encode(self, x):
        x = self.encoder_conv_in(x)
        for down in self.encoder_downs:
            x = down(x)
        for mid in self.encoder_mids:
            x = mid(x)

        x = self.encoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        x, quant_losses = self.quantize(x)
        return x, quant_losses

    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder_conv_in(x)
        for mid in self.decoder_mids:
            x = mid(x)
        for up in self.decoder_ups:
            x = up(x)

        x = self.decoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.decoder_conv_out(x)
        return x

    def forward(self, x: Tensor):
        z, quant_losses = self.encode(x)
        x = self.decode(z)
        return x, z, quant_losses


# VQVAE(3).forward(torch.rand((8, 3, 32, 32)))
