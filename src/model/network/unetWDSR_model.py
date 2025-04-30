
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import math

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.temporal_size = None
        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = in_chans
        scale = 1
        num_blocks = 16    #orginal
        # num_blocks = 4
        num_residual_units = 32
        width_multiplier = 4
        num_outputs = out_chans

        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        for _ in range(num_blocks):
            body.append(
                Block(
                    num_residual_units,
                    kernel_size,
                    width_multiplier,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_blocks),
                ))
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_inputs,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2))
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        if self.temporal_size:
            x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
        x -= self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        x += self.image_mean
        if self.temporal_size:
            x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
        return x


class Block(nn.Module):

    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 width_multiplier=1,
                 weight_norm=torch.nn.utils.weight_norm,
                 res_scale=1):
        super(Block, self).__init__()
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        weight_norm = torch.nn.utils.weight_norm
        body = []
        conv = weight_norm(nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.layers = nn.Sequential(*body)


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)



class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]

        weight_norm = torch.nn.utils.weight_norm
        body = []
        conv_0 = weight_norm(nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1))
        init.ones_(conv_0.weight_g)
        init.zeros_(conv_0.bias)
        body.append(conv_0)
        # self.up_conv += [nn.Sequential(ConvBlock(ch * 2, ch), *body)]
        self.up_conv += [nn.Sequential(ConvBlock(ch * 2, ch), nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1))]
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1 # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1 # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output
