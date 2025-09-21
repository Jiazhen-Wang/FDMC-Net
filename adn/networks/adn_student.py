import torch
import torch.nn as nn
import functools
from copy import deepcopy, copy
from .blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock
# from blocks import ConvolutionBlock, ResidualBlock, FullyConnectedBlock
# from ..utils import print_model, FunctionModel
import torch.nn.functional as F


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


    def forward(self, x):
        identity_input = x
        low_filter1 = x
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        # low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.InstanceNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2*2, ch_2*2 // r_2 , kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2*2 // r_2, ch_2*2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 *2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)
        #g_ori = g
        # spatial attention for cnn branch
        g = torch.cat([g, x], 1)
        g_in = g

        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x = g
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([ x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class Encoder(nn.Module):
    def __init__(self, input_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance'):
        super(Encoder, self).__init__()

        self.conv0 = ConvolutionBlock(
            in_channels=input_ch, out_channels=base_ch, kernel_size=7, stride=1,
            padding=3, pad='reflect', norm=down_norm, activ='relu')

        output_ch = base_ch
        for i in range(1, num_down + 1):
            m = ConvolutionBlock(
                in_channels=output_ch, out_channels=output_ch * 2, kernel_size=4,
                stride=2, padding=1, pad='reflect', norm=down_norm, activ='relu')
            setattr(self, "conv{}".format(i), m)
            output_ch = output_ch * 2

        for i in range(num_residual):
            setattr(self, "res{}".format(i),
                    ResidualBlock(output_ch, pad='reflect', norm=res_norm, activ='relu'))

        self.layers = [getattr(self, "conv{}".format(i)) for i in range(num_down + 1)] + \
                      [getattr(self, "res{}".format(i)) for i in range(num_residual)]

    def forward(self, x):
        # sides = []
        for layer in self.layers:
            x = layer(x)
            # sides.append(x)
        return x

class Frequency_Encoder_low(nn.Module):
    def __init__(self,  input_ch, base_ch, num_down = 2, num_residual = 4, res_norm='instance', down_norm='instance'):
        super(Frequency_Encoder_low, self).__init__()
        self.low_high_fre = dynamic_filter_channel(base_ch)


        self.body = nn.Sequential(nn.Conv2d(input_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False))

        self.encoder_low = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')
        self.encoder_high = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')
        self.encoder_low_artfacts = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')
        self.encoder_high_artfacts = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')

    def forward(self, x):
        res = self.body((x))
        low_part, high_part = self.low_high_fre(res)
        content_low = self.encoder_low(low_part)
        content_high = self.encoder_high(high_part)

        artfacts_low = self.encoder_low_artfacts(low_part)
        artfacts_high = self.encoder_high_artfacts(high_part)
        return content_low, artfacts_low, content_high, artfacts_high


class Frequency_Encoder_high(nn.Module):
    def __init__(self,  input_ch, base_ch, num_down = 2, num_residual = 4, res_norm='instance', down_norm='instance'):
        super(Frequency_Encoder_high, self).__init__()
        self.low_high_fre = dynamic_filter_channel(base_ch)


        self.body = nn.Sequential(nn.Conv2d(input_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False))

        self.encoder_low = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')
        self.encoder_high = Encoder(base_ch, base_ch, num_down, num_residual, res_norm='instance', down_norm='instance')

    def forward(self, x):
        res = self.body((x))
        low_part, high_part = self.low_high_fre(res)
        content_low = self.encoder_low(low_part)
        content_high = self.encoder_high(high_part)
        return content_low,  content_high




import torch
import torch.nn as nn
import torch.nn.functional as F


# Gumbel-Softmax采样函数
def gumbel_softmax_sample(logits, tau=1.0):
    noise = torch.rand_like(logits).uniform_(0, 1)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-8))
    return F.softmax((logits + gumbel_noise) / tau, dim=1)


# SE通道注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 专家模块，包含不同类型的卷积操作
class Expert(nn.Module):
    def __init__(self, in_channels):
        super(Expert, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)  # 空洞卷积
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)  # 深度可分离卷积
        self.se = SEBlock(in_channels)  # 通道注意力

    def forward(self, x):
        # print(x.shape)
        # print(route.shape)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out =  out1 +  out2 + out3  # 动态加权
        return self.se(out)  # 通过注意力模块增强


# 解码器
class Decoder_clean(nn.Module):
    """
    解码器模块，支持动态输入 x, y, z, w 的融合逻辑，并结合 MOE 动态专家模块。
    """

    def __init__(self, output_ch, base_ch, num_up = 2, num_residual = 4, num_sides = 3,  res_norm='instance', up_norm='layer', fuse=True, num_experts=4, tau=1.0):
        """
        初始化 Decoder 模块。

        参数:
        - output_ch: 解码器的最终输出通道数。
        - base_ch: 解码器中卷积的基通道数，逐层变化。
        - num_up: 上采样层的数量。
        - num_residual: 残差块的数量。
        - num_experts: MOE 中的专家数量。
        - tau: Gumbel-Softmax 选择专家时的温度参数。
        - res_norm: 残差块中的归一化类型（例如 'instance' 表示实例归一化）。
        - up_norm: 上采样卷积中的归一化类型（例如 'layer' 表示层归一化）。
        - fuse: 是否将解码器特征与辅助特征进行融合（concat 或 add）。
        """
        super(Decoder_clean, self).__init__()

        self.fuse = fuse
        self.num_experts = num_experts
        self.tau = tau

        input_ch = base_ch * 2 ** num_up
        input_chs = []
        # self.fusion = nn.Conv2d(input_ch * 2, input_ch, 1)

        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual):
            self.residual_blocks.append(
                ResidualBlock(input_ch, pad='reflect', activ='lrelu')
            )
            input_chs.append(input_ch)

        # # 动态专家模块
        # self.experts = nn.ModuleList([Expert(input_ch) for _ in range(self.num_experts)])
        # self.expert_selector = nn.Conv2d(input_ch, num_experts, kernel_size=1)

        # 上采样层
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_up):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    ConvolutionBlock(
                        in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                        stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            )
            input_chs.append(input_ch)
            input_ch //= 2

        # 最终输出层
        self.final_conv = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh'
        )

        # self.fusion = BiFusion_block(ch_1=input_ch, ch_2=input_ch, r_2=4, ch_int=input_ch, ch_out=input_ch)
        self.fusion = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)

    def forward(self, x, y):
        """
        前向传播。

        参数:
        - x, y: 主要输入特征和辅助输入特征。

        返回:
        - 解码后的输出特征或图像。
        """
        if self.fuse:
            # x = self.fusion(torch.cat((x, y), dim=1))
            x = self.fusion(x, y)

        # 通过残差块
        for res_block in self.residual_blocks:
            x = res_block(x)

        # 上采样
        for up_block in self.upsample_blocks:
            x = up_block(x)

        return self.final_conv(x)

# 解码器
class Decoder_artifact(nn.Module):
    """
    解码器模块，支持动态输入 x, y, z, w 的融合逻辑，并结合 MOE 动态专家模块。
    """

    def __init__(self, output_ch, base_ch, num_up = 2, num_residual = 4, num_sides = 3,  res_norm='instance', up_norm='layer', fuse=True, num_experts=4, tau=1.0):
        """
        初始化 Decoder 模块。

        参数:
        - output_ch: 解码器的最终输出通道数。
        - base_ch: 解码器中卷积的基通道数，逐层变化。
        - num_up: 上采样层的数量。
        - num_residual: 残差块的数量。
        - num_experts: MOE 中的专家数量。
        - tau: Gumbel-Softmax 选择专家时的温度参数。
        - res_norm: 残差块中的归一化类型（例如 'instance' 表示实例归一化）。
        - up_norm: 上采样卷积中的归一化类型（例如 'layer' 表示层归一化）。
        - fuse: 是否将解码器特征与辅助特征进行融合（concat 或 add）。
        """
        super(Decoder_artifact, self).__init__()

        self.fuse = fuse
        self.num_experts = num_experts
        self.tau = tau

        input_ch = base_ch * 2 ** num_up
        input_chs = []
        # self.fusion = nn.Conv2d(input_ch * 4, input_ch, 1)

        # 残差块
        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual):
            self.residual_blocks.append(
                ResidualBlock(input_ch, pad='reflect', activ='lrelu')
            )
            input_chs.append(input_ch)

        # # 动态专家模块
        # self.experts = nn.ModuleList([Expert(input_ch) for _ in range(self.num_experts)])
        # self.expert_selector = nn.Conv2d(input_ch, num_experts, kernel_size=1)

        # 上采样层
        self.upsample_blocks = nn.ModuleList()
        for i in range(num_up):
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    ConvolutionBlock(
                        in_channels=input_ch, out_channels=input_ch // 2, kernel_size=5,
                        stride=1, padding=2, pad='reflect', norm=up_norm, activ='lrelu'))
            )
            input_chs.append(input_ch)
            input_ch //= 2

        # 最终输出层
        self.final_conv = ConvolutionBlock(
            in_channels=base_ch, out_channels=output_ch, kernel_size=7,
            stride=1, padding=3, pad='reflect', norm='none', activ='tanh'
        )
        self.fusion = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)
        self.fusion1 = BiFusion_block(ch_1=256, ch_2=256, r_2=4, ch_int=256, ch_out=256)

    def forward(self, x, y, z, t):
        """
        前向传播。

        参数:
        - x, y: 主要输入特征和辅助输入特征。

        返回:
        - 解码后的输出特征或图像。
        """
        if self.fuse:
            # x = self.fusion(torch.cat((x, y, z, t), dim=1))
            x = self.fusion(x,y) + self.fusion1(z,t)

        # 通过残差块
        for res_block in self.residual_blocks:
            x = res_block(x)



        # 上采样
        for up_block in self.upsample_blocks:
            x = up_block(x)

        return self.final_conv(x)




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
        self.encoder_high = Frequency_Encoder_high(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.encoder_low = Frequency_Encoder_low(input_ch, base_ch, num_down, num_residual, res_norm, down_norm)
        self.decoder = Decoder_clean(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse)
        self.decoder_art = Decoder_artifact(input_ch, base_ch, num_down, num_residual, self.n, res_norm, up_norm, fuse)


    def forward1(self, x_low, x_high):

        content_low, sides_low, content_high, sides_high = self.encoder_low(x_low) # encode artifact and content in low-frequency and high-frequency of motion image
        print(content_low.shape)
        content_low_clean, content_high_clean = self.encoder_high(x_high) # encode content of clean image
        # print(content_low.shape, sides_low.shape, content_high.shape, sides_high.shape)

        x1 = self.decoder_art(content_low, content_high, sides_low, sides_high)  #decode image with artifact (motion image self-reconstruction)
        x2 = self.decoder(content_low, content_high) # decode image without artfact (motion image demotion)


        y1 = self.decoder_art(content_low_clean, content_high_clean, sides_low, sides_high)
        y2 = self.decoder(content_low_clean, content_high_clean)

        content_low_clean1, sides_low_clean1, content_high_clean1, sides_high_clean1  = self.encoder_low(y1)
        content_low_clean11, content_high_clean11 = self.encoder_high(x2)

        y2_rec = self.decoder(content_low_clean1, content_high_clean1)

        x2_rec = self.decoder_art(content_low_clean11, content_high_clean11, sides_low_clean1, sides_high_clean1)





        return x1, x2, y1, y2, y2_rec, x2_rec

    # def forward2(self, x_low, x_high):
    #     # print(x_low.shape)
    #     if hasattr(self, "saved") and self.saved[0] is x_low:
    #         sides = self.saved[1]
    #     else:
    #         _, sides = self.encoder_art(x_low)  # encode artifact
    #
    #     code, _ = self.encoder_high(x_high)  # encode high quality image
    #
    #     y1 = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
    #     y2 = self.decoder(code)  # decode without artifact (high quality)
    #     return y1, y2
    #
    # def forward_lh(self, x_low):
    #     code, _ = self.encoder_low(x_low)  # encode low quality image
    #     y = self.decoder(code)
    #     return y
    #
    # def forward_hl(self, x_low, x_high):
    #     _, sides = self.encoder_art(x_low)  # encode artifact
    #     code, _ = self.encoder_high(x_high)  # encode high quality image
    #     y = self.decoder_art(code, sides[-self.n:])  # decode image with artifact (low quality)
    #     return y


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
        sequence.append(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias))
        if norm_layer:
            sequence.append(norm_layer(ndf * nf_mult))
        sequence.append(nn.LeakyReLU(0.2))
        sequence.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))  # output 1 channel prediction map
        # self.out1 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.model = nn.Sequential(*sequence)

    def forward(self, input1):
        """Standard forward."""
        out = self.model(input1)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 256, 256)
    net = dynamic_filter_channel(1)
    a, b = net(x)
    print(a.shape)