import torch
from torch import nn as nn
from torch.nn import functional as F
from argparse import ArgumentParser
from ipdb import set_trace


class UNetAbstract(nn.Module):
    def __init__(self) :
        super().__init__()

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--unet.num_layers', type=int, default=7)
        parser.add_argument('--unet.features_start', type=int, default=64)
        parser.add_argument('--train_bn', action='store_true')
        parser.add_argument('--bilinear', action='store_true')
        parser.add_argument('--inner_normalisation', '-inm', type=str, choices=['InstanceNorm', 'BatchNorm', 'None'], default='InstanceNorm')
        parser.add_argument('--padding_mode', type=str, choices=['zeros', 'reflect'], default='reflect')
        return parser


class UNet(UNetAbstract):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        kwargs :
            bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
            train_bn : Whether to use accumulated batch parameters ( "trained" ) or per batch values
            inner_normalisation : Type of normalisation to use ['InstanceNorm', 'BatchNorm', 'None']
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        **kwargs
    ):
        if num_layers < 1:
            raise ValueError(f'num_layers = {num_layers}, expected: num_layers > 0')

        super().__init__()
        self.num_layers = num_layers
        print(f'Num layers : {self.num_layers} Features Start : {features_start} Padding Mode : {kwargs["padding_mode"]}')

        layers = [DoubleConv(input_channels, features_start, **kwargs)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2, **kwargs))
            feats *= 2
        self.n_hidden_feats = feats

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, **kwargs))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Params :
            x : model input ( b, c, I, J)
        Returns:
            Segmentation : model segmentation ( b, L, I , J)
            Hidden Features : middle hidden representation ( b, ?)
        """
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        hidden_feats = xi[-1]

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1]), hidden_feats

    def get_output_shape(self, in_dim) :
        """
        Return output shapes for this unet structure
        params : in_dim : input dimension ( c, w, h)
        """
        fk_in = torch.zeros((1, in_dim[0], in_dim[1], in_dim[2]))
        output, hidden_feats = self.forward(fk_in)
        return output.shape, hidden_feats.shape


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int, train_bn: bool, inner_normalisation: bool, padding_mode:str, **kwargs):
        super().__init__()
        if inner_normalisation == 'None' :
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
                nn.ReLU(inplace=True)
            )
        else :
            INNER_NORMS = {'InstanceNorm':nn.InstanceNorm2d, 'BatchNorm':nn.BatchNorm2d}
            inm = INNER_NORMS[inner_normalisation]
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
                inm(out_ch, track_running_stats=train_bn),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode=padding_mode),
                inm(out_ch, track_running_stats=train_bn),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch, **kwargs))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool, **kwargs):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
