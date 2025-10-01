from torch import nn
import torch

class Normalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
    def forward(self, x):
        return self.norm(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
        emb_channels=None,
        use_conv=False,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    2 * self.out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            Normalize(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        h = self.in_layers(x)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
                
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h)

        return self.skip_connection(x) + h

class ConditionalUNet(nn.Module):
    def __init__(
            self,
            unet,
            x_dim = 32,
            x_channels = 4,
            feature_channels = 2,
    ):
        super().__init__()
        self.unet = unet
        
        self.fuse_x_in = nn.Conv2d(x_channels, x_channels, 3, padding=1)
        self.fuse_features_in = nn.Conv2d(feature_channels, x_channels, 3, padding=1)

        self.fuse_x_ln = nn.LayerNorm([x_channels, x_dim, x_dim])
        self.fuse_x_features_ln = nn.LayerNorm([x_channels, x_dim, x_dim])

        self.fuse_features = nn.ModuleList([])

        in_ch = 2 * x_channels
        for _ in range(2):
            self.fuse_features.append(
                ResBlock(
                    channels = in_ch,
                    out_channels = x_channels,
                    use_conv=True,
                )
            )

            in_ch = x_channels

        self.fuse_features = nn.Sequential(*self.fuse_features)

    def forward(self, x, x_features, timesteps, encoder_hidden_states=None):
        """
        Fuse the input x and x_features and pass them through the UNet.
        :param x: The input image of shape [B, C, H, W]
        :param x_features: The input features of shape [B, C, H, W]
        :param emb: The embedding.
        :return: The output of the UNet.
        """
        x_dim = x.shape[2:]
        x_features = nn.functional.interpolate(x_features, size=x_dim, mode='nearest')

        x = self.fuse_x_in(x)
        x_features = self.fuse_features_in(x_features)

        x = self.fuse_x_ln(x)
        x_features = self.fuse_x_features_ln(x_features)

        x = torch.cat([x, x_features], dim=1)
        x = self.fuse_features(x)

        return self.unet(x, timesteps, encoder_hidden_states)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)