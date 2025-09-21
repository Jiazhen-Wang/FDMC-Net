
import functools
from copy import deepcopy, copy
from .blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = F.layer_norm(x, x.shape[1:], eps=self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


pad_dict = dict(
     zero = nn.ZeroPad2d,
  reflect = nn.ReflectionPad2d,
replicate = nn.ReplicationPad2d)

conv_dict = dict(
   conv2d = nn.Conv2d,
 deconv2d = nn.ConvTranspose2d)

norm_dict = dict(
     none = lambda x: lambda x: x,
 spectral = lambda x: lambda x: x,
    batch = nn.BatchNorm2d,
 instance = nn.InstanceNorm2d,
    layer = LayerNorm)

activ_dict = dict(
      none = lambda: lambda x: x,
      relu = lambda: nn.ReLU(),
     lrelu = lambda: nn.LeakyReLU(0.2),
     prelu = lambda: nn.PReLU(),
      selu = lambda: nn.SELU(),
      tanh = lambda: nn.Tanh())


class ConvolutionBlock(nn.Module):
    def __init__(self, conv='conv2d', norm='instance', activ='relu', pad='reflect', padding=0, **conv_opts):
        super(ConvolutionBlock, self).__init__()

        self.pad = pad_dict[pad](padding)
        self.conv = conv_dict[conv](**conv_opts)

        out_channels = conv_opts['out_channels']
        self.norm = norm_dict[norm](out_channels)
        if norm == "spectral": self.conv = spectral_norm(self.conv)

        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.conv(self.pad(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm='instance', activ='relu', pad='reflect'):
        super(ResidualBlock, self).__init__()

        block = []
        block.append(ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ=activ, pad=pad))
        block.append(ConvolutionBlock(
            in_channels=channels, out_channels=channels, kernel_size=3,
            stride=1, padding=1, norm=norm, activ='none', pad=pad))
        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x) + x


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_ch, output_ch, norm='none', activ='relu'):
        super(FullyConnectedBlock, self).__init__()

        self.fc = nn.Linear(input_ch, output_ch, bias=True)
        self.norm = norm_dict[norm](output_ch)
        if norm == "spectral": self.fc = spectral_norm(self.fc)
        self.activ = activ_dict[activ]()

    def forward(self, x):
        return self.activ(self.norm(self.fc(x)))

class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter_channel, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group * kernel_size ** 2, group * kernel_size ** 2, kernel_size=1, stride=1,
                                   bias=False)
        self.act_gate = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.ap_2 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter1 = x
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high


class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')
        
        output_ch = base_ch
        for i in range(1, num_down+1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch = output_ch * 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down+1)] + \
            [getattr(self, "res{}".format(i)) for i in range(num_residual)]
        
    def forward(self, x):
        sides = []
        for layer in self.layers:
            x = layer(x)
            sides.append(x)
        return x, sides[::-1]


class Decoder_artfacts(nn.Module):
    """
    解码器模块，支持动态输入 x, y, z, w 的融合逻辑。
    """

    def __init__(self, output_ch, base_ch, num_up = 2, num_residual = 4, num_sides = 3,  res_norm='instance', up_norm='layer', fuse=True):
        """
        初始化 Decoder 模块。

        参数:
        - output_ch: 解码器的最终输出通道数。
        - base_ch: 解码器中卷积的基通道数，逐层变化。
        - num_up: 上采样层的数量。
        - num_residual: 残差块的数量。
        - res_norm: 残差块中的归一化类型（例如 'instance' 表示实例归一化）。
        - up_norm: 上采样卷积中的归一化类型（例如 'layer' 表示层归一化）。
        - fuse: 是否将解码器特征与辅助特征进行融合（concat 或 add）。
        """
        super(Decoder_artfacts, self).__init__()

        # 初始通道数，通常从一个较大的值开始逐步减小
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        # 定义残差块（Residual Blocks）
        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        # 定义上采样层（Upsampling Layers）
        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),  # 最近邻插值，上采样2倍
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))  # 卷积层，减少通道数
            setattr(self, "conv{}".format(i), m)  # 将模块保存到类中
            input_chs.append(input_ch)  # 保存当前输入通道数
            input_ch //= 2  # 更新通道数，减半

        # 最后一层卷积，用于生成输出（通常为图像）
        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')  # 使用 tanh 激活生成[-1, 1]的值
        setattr(self, "conv{}".format(num_up), m)  # 保存最后一层卷积模块
        input_chs.append(base_ch)

        # 将残差块和上采样层的模块存储到 `self.layers` 中
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # 如果需要融合辅助特征（fuse=True）
        self.fusion  = nn.Conv2d(input_ch * 4, input_ch, 1)



    def forward(self, x, y, z=None, w=None):
        """
        前向传播。

        参数:
        - x, y, z, w: 主输入和辅助输入特征。

        返回:
        - 解码后的输出特征或图像。
        """
        x = torch.cat((x, y, z, w), dim = 1)
        m = len(self.layers)

        # 主特征逐层通过解码器的残差块和上采样层（不包含需要融合的层）
        for i in range(m):
            x = self.layers[i](x)  # 融合后的特征经过解码层
        return x


class Decoder_clean(nn.Module):
    """
    解码器模块，支持动态输入 x, y, z, w 的融合逻辑。
    """

    def __init__(self, output_ch, base_ch, num_up = 2, num_residual = 4, num_sides = 3,  res_norm='instance', up_norm='layer', fuse=True):
        """
        初始化 Decoder 模块。

        参数:
        - output_ch: 解码器的最终输出通道数。
        - base_ch: 解码器中卷积的基通道数，逐层变化。
        - num_up: 上采样层的数量。
        - num_residual: 残差块的数量。
        - res_norm: 残差块中的归一化类型（例如 'instance' 表示实例归一化）。
        - up_norm: 上采样卷积中的归一化类型（例如 'layer' 表示层归一化）。
        - fuse: 是否将解码器特征与辅助特征进行融合（concat 或 add）。
        """
        super(Decoder_clean, self).__init__()

        # 初始通道数，通常从一个较大的值开始逐步减小
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        # 定义残差块（Residual Blocks）
        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        # 定义上采样层（Upsampling Layers）
        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),  # 最近邻插值，上采样2倍
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))  # 卷积层，减少通道数
            setattr(self, "conv{}".format(i), m)  # 将模块保存到类中
            input_chs.append(input_ch)  # 保存当前输入通道数
            input_ch //= 2  # 更新通道数，减半

        # 最后一层卷积，用于生成输出（通常为图像）
        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')  # 使用 tanh 激活生成[-1, 1]的值
        setattr(self, "conv{}".format(num_up), m)  # 保存最后一层卷积模块
        input_chs.append(base_ch)

        # 将残差块和上采样层的模块存储到 `self.layers` 中
        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # 如果需要融合辅助特征（fuse=True）
        self.fusion  = nn.Conv2d(input_ch * 2, input_ch, 1)



    def forward(self, x, y):
        """
        前向传播。

        参数:
        - x, y, z, w: 主输入和辅助输入特征。

        返回:
        - 解码后的输出特征或图像。
        """
        x = torch.cat((x, y), dim = 1)
        m = len(self.layers)

        # 主特征逐层通过解码器的残差块和上采样层（不包含需要融合的层）
        for i in range(m):
            x = self.layers[i](x)  # 融合后的特征经过解码层
        return x

class Decoder(nn.Module):
    def __init__(self, output_ch, base_ch, num_up, num_residual, num_sides, res_norm='instance', up_norm='layer',
                 fuse=False):
        super(Decoder, self).__init__()
        input_ch = base_ch * 2 ** num_up
        input_chs = []

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(input_ch, pad='reflect', norm=res_norm, activ='lrelu'))
            input_chs.append(input_ch)

        for i in range(num_up):
            m = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvolutionBlock(
                    in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                    stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            setattr(self, "conv{}".format(i), m)
            input_chs.append(input_ch)
            input_ch = input_ch // 2

        m = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh')
        setattr(self, "conv{}".format(num_up), m)
        input_chs.append(base_ch)

        self.layers = [getattr(self, "res{}".format(i)) for i in range(num_residual)] + \
                      [getattr(self, "conv{}".format(i)) for i in range(num_up + 1)]

        # If true, fuse (concat and conv) the side features with decoder features
        # Otherwise, directly add artifact feature with decoder features
        if fuse:
            # print('2222')
            input_chs = input_chs[-num_sides:]
            for i in range(num_sides):
                setattr(self, "fuse{}".format(i),
                        nn.Conv2d(input_chs[i] * 2, input_chs[i], 1))
            self.fuse = lambda x, y, i: getattr(self, "fuse{}".format(i))(torch.cat((x, y), 1))
        else:
            # print('1111')
            self.fuse = lambda x, y, i: x + y

    def forward(self, x, sides=[]):
        m, n = len(self.layers), len(sides)
        assert m >= n, "Invalid side inputs"

        for i in range(m - n):
            # print('111')
            x = self.layers[i](x)
            # print(x.shape)

        for i, j in enumerate(range(m - n, m)):
            x = self.fuse(x, sides[i], i)
            x = self.layers[j](x)
        return x

class ADN(nn.Module):
    """
    Image with artifact is denoted as low quality image
    Image without artifact is denoted as high quality image
    """

    def __init__(self, input_ch=1, base_ch=64, num_down=2, num_residual=4, num_sides="all",
        res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False):
        super(ADN, self).__init__()

        self.n = num_down + num_residual + 1 if num_sides == "all" else num_sides
        self.encoder_low = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_high = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_art = Encoder(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.decoder = Decoder(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse)
        self.decoder_art = self.decoder if shared_decoder else deepcopy(self.decoder)

    def forward1(self, x_low):
        # print(x_low.shape)
        _, sides = self.encoder_art(x_low)  # encode artifact
        # print(sides[-self.n:].shape)
        self.saved = (x_low, sides)
        code, _ = self.encoder_low(x_low)  # encode low quality image
        print(code.shape)
        y1 = self.decoder_art(code, sides[-self.n:]) # decode image with artifact (low quality)
        y2 = self.decoder(code) # decode image without artifact (high quality)
        return y1, y2

    def forward2(self, x_low, x_high):
        # print(x_low.shape)
        if hasattr(self, "saved") and self.saved[0] is x_low: sides = self.saved[1]
        else: _, sides = self.encoder_art(x_low)  # encode artifact

        code, _ = self.encoder_high(x_high) # encode high quality image

        y1 = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
        y2 = self.decoder(code) # decode without artifact (high quality)
        return y1, y2

    def forward_lh(self, x_low):
        code, _ = self.encoder_low(x_low)  # encode low quality image
        y = self.decoder(code)
        return y

    def forward_hl(self, x_low, x_high):
        _, sides = self.encoder_art(x_low)  # encode artifact
        code, _ = self.encoder_high(x_high) # encode high quality image
        y = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
        return y


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
    
    This class is adopted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) is str:
            norm_layer = {
                "layer": nn.LayerNorm,
                "instance": nn.InstanceNorm2d,
                "batch": nn.BatchNorm2d,
              "none": None}[norm_layer]

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = []
        sequence.append(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw))
        sequence.append(nn.LeakyReLU(0.2))
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence.append(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias))
            if norm_layer:
                sequence.append(norm_layer(ndf * nf_mult))
            sequence.append(nn.LeakyReLU(0.2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        if norm_layer:
            sequence.append(norm_layer(ndf * nf_mult))
        sequence.append(nn.LeakyReLU(0.2))
        sequence.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))  # output 1 channel prediction map
        # self.out1 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.model = nn.Sequential(*sequence)

    def forward(self, input1):
        """Standard forward."""
        out = self.model(input1)
        return out


if __name__ == '__main__':
    x=torch.randn(1,1,256,256)
    net=dynamic_filter_channel(1)
    a, b = net(x)
    print(a.shape)
