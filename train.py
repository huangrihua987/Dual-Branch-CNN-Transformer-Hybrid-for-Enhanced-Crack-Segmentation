import torch.nn as nn


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


# pw dw
class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


import torch
import torch.nn as nn


# 基本的Conv + BN + GELU模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = DSC(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 构建整个多分支网络
class MultiBranchNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiBranchNetwork, self).__init__()

        # 分支1：卷积核大小为1的 Conv + BN + GELU 模块
        self.branch1 = ConvBlock(in_channels, out_channels, kernel_size=1)

        # 分支2：卷积核大小为3的 Conv + BN + GELU 模块
        self.branch2 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)

        # 分支4：对分支2的输出再进行卷积核大小为1的 Conv + BN + GELU 模块
        self.branch4 = ConvBlock(out_channels, out_channels, kernel_size=1)

        # 分支5：对分支2的输出进行卷积核大小为3，再进行5x5的 Conv + BN + GELU 模块
        self.branch5_1 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.branch5_2 = ConvBlock(out_channels, out_channels, kernel_size=5, padding=2)

        # 分支3：卷积核大小为5的 Conv + BN + GELU 模块
        self.branch3 = ConvBlock(in_channels, out_channels, kernel_size=5, padding=2)

        # 分支6：对分支3的输出进行卷积核大小为1的 Conv + BN + GELU 模块
        self.branch6 = ConvBlock(out_channels, out_channels, kernel_size=1)

        # 分支7：对分支3的输出进行5x5卷积，再进行7x7卷积
        self.branch7_1 = ConvBlock(out_channels, out_channels, kernel_size=5, padding=2)
        self.branch7_2 = ConvBlock(out_channels, out_channels, kernel_size=7, padding=3)

        # 分支8：对分支5的输出进行1x1卷积
        self.branch8 = ConvBlock(out_channels, out_channels, kernel_size=1)

        # 分支9：对分支5的输出进行3x3卷积，再进行7x7卷积
        self.branch9_1 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.branch9_2 = ConvBlock(out_channels, out_channels, kernel_size=7, padding=3)

        # 分支10：对分支9的输出进行1x1卷积
        self.branch10 = ConvBlock(out_channels, out_channels, kernel_size=1)

        # 分支11：对分支9的输出进行3x3卷积，再进行9x9卷积
        self.branch11_1 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.branch11_2 = ConvBlock(out_channels, out_channels, kernel_size=9, padding=4)

        # 分支12：对分支7的输出进行1x1卷积
        self.branch12 = ConvBlock(out_channels, out_channels, kernel_size=1)

        # 分支13：对分支7的输出进行5x5卷积，再进行9x9卷积
        self.branch13_1 = ConvBlock(out_channels, out_channels, kernel_size=5, padding=2)
        self.branch13_2 = ConvBlock(out_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x):
        # 分支1：保持不动
        out1 = self.branch1(x)

        # 分支2：3x3卷积
        out2 = self.branch2(x)

        # 分支4：对分支2的输出进行1x1卷积
        out4 = self.branch4(out2)

        # 分支5：对分支2的输出进行3x3卷积，再进行5x5卷积
        out5 = self.branch5_1(out2)
        out5 = self.branch5_2(out5)

        # 分支8：对分支5的输出进行1x1卷积
        out8 = self.branch8(out5)

        # 分支9：对分支5的输出进行3x3卷积，再进行7x7卷积
        out9 = self.branch9_1(out5)
        out9 = self.branch9_2(out9)

        # 分支10：对分支9的输出进行1x1卷积
        out10 = self.branch10(out9)

        # 分支11：对分支9的输出进行3x3卷积，再进行9x9卷积
        out11 = self.branch11_1(out9)
        out11 = self.branch11_2(out11)

        # 分支3：5x5卷积
        out3 = self.branch3(x)

        # 分支6：对分支3的输出进行1x1卷积
        out6 = self.branch6(out3)

        # 分支7：对分支3的输出进行5x5卷积，再进行7x7卷积
        out7 = self.branch7_1(out3)
        out7 = self.branch7_2(out7)

        # 分支12：对分支7的输出进行1x1卷积
        out12 = self.branch12(out7)

        # 分支13：对分支7的输出进行5x5卷积，再进行9x9卷积
        out13 = self.branch13_1(out7)
        out13 = self.branch13_2(out13)

        # 最终将分支1、4、8、10、11、6、12、13的结果在通道维度上进行拼接
        out = torch.cat([out1, out4, out8, out10, out11, out6, out12, out13], dim=1)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-scale Attention Network for Single Image Super-Resolution (CVPR 2024)
# https://arxiv.org/abs/2209.14145

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


import torch.nn.functional as F
import torch
import torch.nn as nn


class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.n_feats = n_feats
        self.norm = nn.BatchNorm2d(n_feats)
        self.need_padding = n_feats % 3 != 0

        # 如果通道数不能被3整除，则需要进行填充
        if self.need_padding:
            self.padding_channels = 3 - (n_feats % 3)
        else:
            self.padding_channels = 0

        self.LKA = MLKA(n_feats + self.padding_channels)
        self.LFE = GSAU(n_feats + self.padding_channels)

    def forward(self, x):
        x = self.norm(x)

        # 如果需要进行填充以适应通道数
        if self.need_padding:
            padding = torch.zeros(x.size(0), self.padding_channels, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, padding], dim=1)

        x = self.LKA(x)
        x = self.LFE(x)

        # 移除填充的通道
        if self.need_padding:
            x = x[:, :-self.padding_channels, :, :]

        return x


class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # 不再要求 n_feats 必须整除 3
        self.part1 = n_feats // 3
        self.part2 = (n_feats - self.part1) // 2
        self.part3 = n_feats - self.part1 - self.part2

        self.lka_part1 = nn.Conv2d(self.part1, self.part1, kernel_size=3, padding=1)
        self.lka_part2 = nn.Conv2d(self.part2, self.part2, kernel_size=5, padding=2)
        self.lka_part3 = nn.Conv2d(self.part3, self.part3, kernel_size=7, padding=3)

    def forward(self, x):
        # 将通道分割为三部分
        x1, x2, x3 = torch.split(x, [self.part1, self.part2, self.part3], dim=1)
        # 分别通过不同的卷积核
        out1 = self.lka_part1(x1)
        out2 = self.lka_part2(x2)
        out3 = self.lka_part3(x3)
        # 将结果拼接
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = nn.BatchNorm2d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # layer1
        self.conv1_1 = DSC(3, 32)
        self.conv1_2 = DSC(32, 32)
        self.norm1 = nn.BatchNorm2d(32)
        self.act = nn.GELU()

        self.c1_1 = nn.Sequential(self.conv1_1,
                                  self.norm1,
                                  self.act)
        self.c1_2 = nn.Sequential(self.conv1_2,
                                  self.norm1,
                                  self.act)
        self.res1 = nn.Conv2d(288, 32, kernel_size=1)  # 修改后的
        self.multi_branch = MultiBranchNetwork(32, 32)
        self.mab = MAB(32)
        self.pool1 = DSC(32, 32, 2, 2, 0)

        # layer2
        self.conv2_1 = DSC(32, 64)
        self.conv2_2 = DSC(64, 64)
        self.norm2 = nn.BatchNorm2d(64)
        self.c2_1 = nn.Sequential(self.conv2_1,
                                  self.norm2,
                                  self.act)
        self.c2_2 = nn.Sequential(self.conv2_2,
                                  self.norm2,
                                  self.act)
        self.res2 = nn.Conv2d(576, 64, kernel_size=1)  # 修改后的
        self.multi_branch2 = MultiBranchNetwork(64, 64)
        self.mab = MAB(64)
        self.pool2 = DSC(64, 64, 2, 2, 0)

        # layer3
        self.conv3_1 = DSC(64, 128)
        self.conv3_2 = DSC(128, 128)
        self.norm3 = nn.BatchNorm2d(128)
        self.c3_1 = nn.Sequential(self.conv3_1,
                                  self.norm3,
                                  self.act)
        self.c3_2 = nn.Sequential(self.conv3_2,
                                  self.norm3,
                                  self.act)
        self.res3 = nn.Conv2d(1152, 128, kernel_size=1)  # 修改后的
        self.multi_branch3 = MultiBranchNetwork(128, 128)
        self.mab = MAB(128)
        self.pool3 = DSC(128, 128, 2, 2, 0)

        # layer4
        self.conv4_1 = DSC(128, 256)
        self.conv4_2 = DSC(256, 256)
        self.norm4 = nn.BatchNorm2d(256)
        self.c4_1 = nn.Sequential(self.conv4_1,
                                  self.norm4,
                                  self.act)
        self.c4_2 = nn.Sequential(self.conv4_2,
                                  self.norm4,
                                  self.act)
        self.res4 = nn.Conv2d(2304, 256, kernel_size=1)  # 修改后的
        self.multi_branch4 = MultiBranchNetwork(256, 256)
        self.mab = MAB(256)
        self.pool4 = DSC(256, 256, 2, 2, 0)

        self.pool5 = DSC(256, 512, 2, 2, 0)

    def forward(self, x):
        # layer1
        x1_1 = self.c1_1(x)
        x1_p = self.pool1(x1_1)
        x1_out = self.multi_branch(x1_p)

        temp1 = torch.cat([x1_p, x1_out], dim=1)
        x1_out = self.res1(temp1)

        # layer2
        x2_1 = self.c2_1(x1_out)
        x2_p = self.pool2(x2_1)
        x2_out = self.multi_branch2(x2_p)
        # x2_out = self.c2_2(x2_p)
        temp2 = torch.cat([x2_p, x2_out], dim=1)
        x2_out = self.res2(temp2)

        # layer3
        x3_1 = self.c3_1(x2_out)
        x3_p = self.pool3(x3_1)
        x3_out = self.multi_branch3(x3_p)
        # x3_out = self.c3_2(x3_p)
        temp3 = torch.cat([x3_p, x3_out], dim=1)
        x3_out = self.res3(temp3)

        # layer4
        x4_1 = self.c4_1(x3_out)
        x4_p = self.pool4(x4_1)
        x4_out = self.multi_branch4(x4_p)
        # x4_out = self.c4_2(x4_p)
        temp4 = torch.cat([x4_p, x4_out], dim=1)
        x4_out = self.res4(temp4)

        # layer5
        out = self.pool5(x4_out)

        return x1_out, x2_out, x3_out, x4_out, out


import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, dim, p_size):
        super().__init__()
        self.embed = DSC(3, dim, p_size, p_size, 0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.norm(self.embed(x))
        return x


class PatchMerge(nn.Module):
    def __init__(self, inc, outc, kernel_size=2):
        super().__init__()
        self.merge = DSC(inc, outc, k_size=kernel_size, stride=kernel_size, padding=0)
        self.norm = nn.BatchNorm2d(outc)

    def forward(self, x):
        return self.norm(self.merge(x))


class Attention(nn.Module):
    def __init__(self, dim, window_size=2, num_head=8, qk_scale=None, qkv_bias=None, alpha=0.5):
        super().__init__()
        head_dim = int(dim / num_head)
        self.dim = dim

        self.l_head = int(num_head * alpha)
        self.l_dim = self.l_head * head_dim

        self.h_head = num_head - self.l_head
        self.h_dim = self.h_head * head_dim

        self.ws = window_size
        if self.ws == 1:
            self.h_head = 0
            self.h_dim = 0
            self.l_head = num_head
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.l_head > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = DSC(self.dim, self.l_dim)
            self.l_kv = DSC(self.dim, self.l_dim * 2)
            self.l_proj = DSC(self.l_dim, self.l_dim)

        if self.h_head > 0:
            self.h_qkv = DSC(self.dim, self.h_dim * 3)
            self.h_proj = DSC(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        qkv = self.h_qkv(x).reshape(B, 3, self.h_head, self.h_dim // self.h_head, total_groups, self.ws * self.ws) \
            .permute(1, 0, 4, 2, 5, 3)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim).permute(0, 3, 1, 2)
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, C, H, W = x.shape
        q = self.l_q(x).reshape(B, self.l_head, self.l_dim // self.l_head, H * W).permute(0, 1, 3, 2)

        if self.ws > 1:
            x_ = self.sr(x)
            kv = self.l_kv(x_).reshape(B, 2, self.l_head, self.l_dim // self.l_head, -1).permute(1, 0, 2, 4, 3)
        else:
            kv = self.l_kv(x).reshape(B, 2, self.l_head, self.l_dim // self.l_head, -1).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim).permute(0, 3, 1, 2)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        if self.h_head > 0 and self.l_head > 0:
            x_h = self.hifi(x)
            x_l = self.lofi(x)
            x = torch.cat([x_h, x_l], dim=1)
            return x

        elif self.l_head > 0 and self.h_head == 0:
            x_l = self.lofi(x)
            return x_l

        else:
            x_h = self.hifi(x)
            return x_h


import torch
import torch.nn as nn


# 论文地址：https://arxiv.org/pdf/2308.03364
# 论文：Dual Aggregation Transformer for Image Super-Resolution, ICCV 2023
class SpatialGate(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, C, H, W), H, W
        Output: x: (B, H*W, C)
        """
        # 调整形状
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        # 恢复形状
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self, inc, outc=None, dropout=0.2):
        super().__init__()
        # outc = outc or inc * 2
        outc = outc or inc
        self.fc1 = nn.Conv2d(inc, outc, 1)
        self.fc2 = DSC(outc, outc)
        self.fc3 = nn.Conv2d(outc, inc, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x


class Block(nn.Module):
    def __init__(self, inc, window_size=2, num_head=8, alpha=0.5, dropout=0.):
        super().__init__()
        self.norm = nn.BatchNorm2d(inc)

        self.HiLo = Attention(inc, window_size=window_size, num_head=num_head, alpha=alpha)
        self.mlp = Mlp(inc, dropout=dropout)

    def forward(self, x):
        # 计算残差连接并传递高度和宽度参数
        x = x + self.norm(self.HiLo(x))
        x = x + self.norm(self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # layer1
        self.embed = PatchEmbed(64, 4)
        self.block1 = Block(64, window_size=2, num_head=8, alpha=0.4, dropout=0.)
        self.mab1 = MAB(64)

        # layer2
        self.merge2 = PatchMerge(64, 128)
        self.block2 = Block(128, window_size=2, num_head=8, alpha=0.3, dropout=0.)
        self.mab2 = MAB(128)

        # layer3
        self.merge3 = PatchMerge(128, 256)
        self.block3 = Block(256, window_size=2, num_head=8, alpha=0.2, dropout=0.)
        self.mab3 = MAB(256)

        # layer4
        self.merge4 = PatchMerge(256, 512)
        self.block4 = Block(512, window_size=2, num_head=8, alpha=0.1, dropout=0.)
        self.mab4 = MAB(512)

    def forward(self, x):
        B, C, H, W = x.shape

        # layer1
        x1 = self.embed(x)
        x1 = self.block1(x1)
        x1 = self.mab1(x1)

        # layer2
        x2 = self.merge2(x1)
        x2 = self.block2(x2)
        x2 = self.mab2(x2)

        # layer3
        x3 = self.merge3(x2)
        x3 = self.block3(x3)
        x3 = self.mab3(x3)

        # layer4
        x4 = self.merge4(x3)
        out = self.block4(x4)
        out = self.mab4(out)

        return x1, x2, x3, out


import torch
import torch.nn as nn


class FCM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(dim, dim)
        self.down = IDSC(3 * dim, dim)

        self.fuse = nn.Sequential(IDSC(3 * dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  DSC(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU(),
                                  DSC(dim, dim),
                                  nn.BatchNorm2d(dim),
                                  nn.GELU()
                                  )

    def forward(self, x1, y1):
        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = y1.shape

        x_temp = self.avg(x1)
        y_temp = self.avg(y1)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = x1.permute(0, 2, 3, 1)
        y_temp = y1.permute(0, 2, 3, 1)

        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        out1 = torch.cat([x1, y1], dim=3)

        out2 = x1 * y1

        fuse = torch.cat([out1, out2], dim=3)
        fuse = fuse.permute(0, 3, 1, 2)

        out = self.fuse(fuse)
        out = out + self.down(fuse)

        return out


import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.t3 = IDSC(384, 256)
        self.t2 = IDSC(192, 128)
        self.t1 = IDSC(96, 64)

        self.block3 = Block(256, window_size=2, alpha=0.2)
        self.block2 = Block(128, window_size=2, alpha=0.3)
        self.block1 = Block(64, window_size=2, alpha=0.4)

        self.up = nn.PixelShuffle(2)

        self.final = nn.Sequential(nn.PixelShuffle(4),
                                   IDSC(4, 1))

    def forward(self, x, x1, x2, x3):
        # 第一层上采样并传递 H 和 W
        temp = self.up(x)
        B, C, H, W = temp.shape  # 获取高度和宽度
        temp = torch.cat([temp, x3], dim=1)
        temp = self.t3(temp)
        x3_out = self.block3(temp)  # 传递 H 和 W

        # 第二层上采样并传递 H 和 W
        temp = self.up(x3_out)
        B, C, H, W = temp.shape  # 更新 H 和 W
        temp = torch.cat([temp, x2], dim=1)
        temp = self.t2(temp)
        x2_out = self.block2(temp)  # 传递 H 和 W

        # 第三层上采样并传递 H 和 W
        temp = self.up(x2_out)
        B, C, H, W = temp.shape  # 更新 H 和 W
        temp = torch.cat([temp, x1], dim=1)
        temp = self.t1(temp)
        x1_out = self.block1(temp)  # 传递 H 和 W

        # 最后输出层
        out = self.final(x1_out)

        return out


import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(512, 256, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(512, 256, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(256, 512, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        # Channel attention for f1
        avg_1 = torch.mean(f1, dim=(2, 3), keepdim=True)
        max_1 = torch.amax(f1, dim=(2, 3), keepdim=True)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1)
        max_1 = self.max11(max_1)
        a1 = avg_1 + max_1

        # Channel attention for f2
        avg_2 = torch.mean(f2, dim=(2, 3), keepdim=True)
        max_2 = torch.amax(f2, dim=(2, 3), keepdim=True)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2)
        max_2 = self.max22(max_2)
        a2 = avg_2 + max_2

        # Cross attention between f1 and f2
        cross = torch.matmul(a1.view(b, c, -1), a2.view(b, c, -1).transpose(1, 2))

        f1_weight = F.softmax(cross, dim=-1)
        f2_weight = F.softmax(cross.transpose(1, 2), dim=-1)

        f1_flat = f1.view(b, c, -1)
        f2_flat = f2.view(b, c, -1)

        f1_attended = torch.matmul(f1_weight, f1_flat).view(b, c, h, w)
        f2_attended = torch.matmul(f2_weight, f2_flat).view(b, c, h, w)

        # Spatial attention for f1
        avg_out_f1 = torch.mean(f1_attended, dim=1, keepdim=True)
        max_out_f1 = torch.amax(f1_attended, dim=1, keepdim=True)
        f1_spatial = torch.cat([avg_out_f1, max_out_f1], dim=1)
        f1_spatial = F.relu(self.conv1_spatial(f1_spatial))
        f1_spatial = self.conv2_spatial(f1_spatial)
        f1_spatial = F.softmax(f1_spatial.view(b, -1), dim=-1).view(b, 1, h, w)

        # Spatial attention for f2
        avg_out_f2 = torch.mean(f2_attended, dim=1, keepdim=True)
        max_out_f2 = torch.amax(f2_attended, dim=1, keepdim=True)
        f2_spatial = torch.cat([avg_out_f2, max_out_f2], dim=1)
        f2_spatial = F.relu(self.conv1_spatial(f2_spatial))
        f2_spatial = self.conv2_spatial(f2_spatial)
        f2_spatial = F.softmax(f2_spatial.view(b, -1), dim=-1).view(b, 1, h, w)

        # Apply spatial attention
        f1 = f1 * f1_spatial + f1
        f2 = f2 * f2_spatial + f2

        return f1, f2


import torch
import torch.nn as nn


# 论文：CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation
# 论文地址：https://arxiv.org/pdf/2405.10530


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2, x4):
        x_fused = torch.cat([x1, x2, x4], dim=1)
        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels * 3, out_channels)

    def forward(self, x1, x2, x4, last=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2, x4)
        return x_fused


import torch
import torch.nn as nn
import math


# 论文：ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation(IMAVIS)
# 论文地址：https://arxiv.org/abs/2312.06458

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # 自适应平均池化
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # 1D卷积
        y = self.sigmoid(y)  # Sigmoid激活
        return x * y.expand_as(x)  # 通道逐元素相乘


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


# Channel and Position Attention Mechanism (CPAM)
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1, input2 = x[0], x[1]
        input1 = self.channel_att(input1)
        x = input1 + input2
        x = self.local_att(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = CNN()
        self.encoder2 = Transformer()
        self.cafm = CAFM()
        self.fuse1 = CPAM(64)
        self.fuse2 = CPAM(128)
        self.fuse3 = CPAM(256)

        self.Conv = nn.Sequential(IDSC(1024, 512),
                                  nn.BatchNorm2d(512),
                                  nn.GELU())
        self.decoder = Decoder()

        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x1, x2, x3, x4, out1 = self.encoder1(x)
        y1, y2, y3, out2 = self.encoder2(x)

        # out1,out2 = self.cafm(out1, out2)

        # 使用 CPAM 模块替代 MSAA
        f1 = self.fuse1(x2, y1)  # 传递 x2 和 y1
        f2 = self.fuse2(x3, y2)  # 传递 x3 和 y2
        f3 = self.fuse3(x4, y3)  # 传递 x4 和 y3

        B1, C1, H1, W1 = out1.shape
        B2, C2, H2, W2 = out2.shape
        x_temp = self.avg(out1)
        y_temp = self.avg(out2)
        x_weight = self.linear(x_temp.reshape(B1, 1, 1, C1))
        y_weight = self.linear(y_temp.reshape(B2, 1, 1, C2))
        x_temp = out1.permute(0, 2, 3, 1)
        y_temp = out2.permute(0, 2, 3, 1)
        x1 = x_temp * x_weight
        y1 = y_temp * y_weight

        x1 = x1.permute(0, 3, 1, 2)
        y1 = y1.permute(0, 3, 1, 2)

        out = torch.cat([x1, y1], dim=1)
        out = self.Conv(out)

        mask = self.decoder(out, f1, f2, f3)

        return mask


import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np


class Datases_loader(Dataset):
    def __init__(self, root_images, root_masks, h, w):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            self.images.append(img_file)
            self.labels.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]
        image = Image.open(image)
        mask = Image.open(mask)
        tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(16, fill=144),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        image = image.convert('RGB')
        # image = image.filter(ImageFilter.SHARPEN)
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        seed = np.random.randint(1459343089)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        img = tf(image)
        img = norm(img)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        mask = tf(mask)
        mask[mask > 0] = 1.

        sample = {'image': img, 'mask': mask, }

        return sample


if __name__ == '__main__':
    imgdir = r'/kaggle/input/crackforest4/kaggle/working/train/images'
    labdir = r'/kaggle/input/crackforest4/kaggle/working/train/masks'
    d = Datases_loader(imgdir, labdir, 512, 512)
    d = Datases_loader(imgdir, labdir, 512, 512)
    my_dataloader = DataLoader(d, batch_size=8, shuffle=False)
    # imshow_image(mydata_loader)
    # save_image(mydata_loader)

import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Loss, self).__init__()

    def forward(self, logits, targets):
        p = logits.view(-1, 1)
        t = targets.view(-1, 1)
        loss1 = F.binary_cross_entropy_with_logits(p, t, reduction='mean')

        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss2 = 1 - score.sum() / num

        return loss2 + loss1


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader


def compute_confusion_matrix(precited, expected):
    # part = precited ^ expected
    part = np.logical_xor(precited, expected)
    pcount = np.bincount(part)
    # tp_list = list(precited & expected)
    # fp_list = list(precited & ~expected)
    tp_list = list(np.logical_and(precited, expected))
    fp_list = list(np.logical_and(precited, np.logical_not(expected)))
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp
    return tp, fp, tn, fn


def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = (2 * precision * recall) / (precision + recall)
    miou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return accuracy, precision, recall, F1, miou


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 2
lr = 0.001
items = 80

model = Net().to(device)
criterion = Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=15, T_mult=1)

savedir = r'/kaggle/working/model'
if not os.path.exists(savedir):
    os.makedirs(savedir)

imgpath = r'/kaggle/input/crackforest4/kaggle/working/train/images'
labpath = r'/kaggle/input/crackforest4/kaggle/working/train/masks'
imgsz = 512

dataset = Datases_loader(imgpath, labpath, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)

ls_loss = []


def train():
    for epoch in range(items):
        lossx = 0
        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, lab)
            loss.backward()
            optimizer.step()

            lossx += loss

        scheduler.step()
        lossx = lossx / dataset.num_of_samples()
        ls_loss.append(lossx.item())

        print(f"Epoch {epoch + 1}/{items}, Loss: {lossx:.4f}")

    torch.save(model.state_dict(), os.path.join(savedir, 'model.pth'))


if __name__ == '__main__':
    train()

    # 将最终的 loss 写入文件
    output_str = 'loss:' + str(ls_loss) + '\n'
    filename = r'/kaggle/working/output.txt'
    with open(filename, mode='w', newline='') as f:
        f.writelines(output_str)
        print(f"File written successfully to {filename}")