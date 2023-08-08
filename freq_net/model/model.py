import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from freq_net.model.two_stage_transforms import TwoStageDCT

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")
from freq_net.base import BaseModel


# TODO: convolution args and group depth size


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        groups=groups,
    )


class ResidualBlock(nn.Module):
    def __init__(
        self, depthwise, conv, n_feat, kernel_size, bias=True, act=nn.LeakyReLU(True)
    ):
        super(ResidualBlock, self).__init__()

        if depthwise:
            conv1 = conv(n_feat, n_feat, kernel_size, groups=n_feat, bias=bias)
        else:
            conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        conv2 = conv(n_feat, n_feat, kernel_size, bias=bias)

        self.body = nn.Sequential(conv1, act, conv2)

        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


# num Residual Group = 10 = n_resblocks (paper)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act, n_resblocks=10, depthwise=False):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResidualBlock(
                depthwise=depthwise,
                conv=conv,
                n_feat=n_feat,
                kernel_size=kernel_size,
                bias=True,
                act=act,
            )
            for _ in range(n_resblocks)
        ]

        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # short skip connection
        res = self.body(x)
        res += x
        return res


# num res groups = 7,
class SEN(nn.Module):
    def __init__(
        self,
        n_resgroups=7,
        n_deformable_resgroups=3,
        n_resblocks=10,
        n_feat=1,
        conv=default_conv,
    ):
        super(SEN, self).__init__()

        kernel_size = 3
        act = nn.LeakyReLU(True)

        conv1 = conv(n_feat, n_feat, kernel_size)

        normal_res_groups = [
            ResidualGroup(
                conv=conv,
                n_feat=n_feat,
                kernel_size=kernel_size,
                act=act,
                n_resblocks=n_resblocks,
                depthwise=False,
            )
            for _ in range(n_resgroups)
        ]
        shrinking_trunk = [
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(64, 100, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(100, 100, kernel_size=3, stride=2, padding=1),
        ]
        # deformable_res_groups = [
        #     ResidualGroup(
        #         conv=conv,
        #         n_feat=n_feat,
        #         kernel_size=kernel_size,
        #         act=act,
        #         n_resblocks=n_resblocks,
        #         depthwise=True,
        #     )
        #     for _ in range(n_deformable_resgroups)
        # ]
        modules_body = [conv1, *normal_res_groups, *shrinking_trunk]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


# num res groups = 7, num depthwise res groups = 3
class FRN(nn.Module):
    def __init__(
        self,
        n_resgroups=7,
        n_depthwise_resgroups=3,
        n_resblocks=10,
        n_feat=100,
        conv=default_conv,
    ):
        super(FRN, self).__init__()

        kernel_size = 3
        act = nn.LeakyReLU(True)

        depthwise_res_groups = [
            ResidualGroup(
                conv=conv,
                n_feat=n_feat,
                kernel_size=kernel_size,
                act=act,
                n_resblocks=n_resblocks,
                depthwise=True,
            )
            for _ in range(n_depthwise_resgroups)
        ]
        normal_res_groups = [
            ResidualGroup(
                conv=conv,
                n_feat=n_feat,
                kernel_size=kernel_size,
                act=act,
                n_resblocks=n_resblocks,
                depthwise=False,
            )
            for _ in range(n_resgroups)
        ]
        modules_body = [*depthwise_res_groups, *normal_res_groups]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # x = self.sub_mean(x)

        res = self.body(x)
        res += x

        return x


class FreqNet(nn.Module):
    def __init__(self, is_test=False):
        super(FreqNet, self).__init__()
        # conv=default_conv
        # n_resgroups = args.n_resgroups
        # n_depthwise_resgroups = args.n_depthwise_resgroups
        # n_resblocks = args.n_resblocks
        # n_feat = args.n_feat
        # kernel_size = 3
        # # reduction = args.reduction
        # # scale = args.scale[0]
        # act = nn.LeakyReLU(True)
        self.frn = FRN()
        self.sen = SEN()
        self.is_test = is_test
        self.transform = TwoStageDCT()

        self.weight1 = 0.5
        self.weight2 = 0.5

    def forward(self, img_s, img_dct):
        # x = self.sub_mean(x)
        feature_maps, normalized_feature_maps = self.transform.two_stage_dct_in(img_dct)
        lower = self.frn(normalized_feature_maps)

        upper = self.sen(img_s)

        out = upper * self.weight1 + lower * self.weight2

        if not self.is_test:
            return out, None

        hr_image = self.transform.two_stage_idct_out(img_s, img_dct, feature_maps, out)
        return out, hr_image  # for MetrickTracker at test time


# class FreqNet(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(FreqNet, self).__init__()

#         n_resgroups = args.n_resgroups
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         kernel_size = 3
#         reduction = args.reduction
#         scale = args.scale[0]
#         act = nn.ReLU(True)

#         # RGB mean for DIV2K
#         rgb_mean = (0.4488, 0.4371, 0.4040)
#         rgb_std = (1.0, 1.0, 1.0)
#         self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

#         # define head module
#         modules_head = [conv(args.n_colors, n_feats, kernel_size)]

#         # define body module
#         modules_body = [
#             ResidualGroup(
#                 conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
#             for _ in range(n_resgroups)]

#         modules_body.append(conv(n_feats, n_feats, kernel_size))

#         # define tail module
#         modules_tail = [
#             common.Upsampler(conv, scale, n_feats, act=False),
#             conv(n_feats, args.n_colors, kernel_size)]

#         self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

#         self.head = nn.Sequential(*modules_head)
#         self.body = nn.Sequential(*modules_body)
#         self.tail = nn.Sequential(*modules_tail)

#     def forward(self, x):
#         x = self.sub_mean(x)
#         x = self.head(x)

#         res = self.body(x)
#         res += x

#         x = self.tail(res)
#         x = self.add_mean(x)

#         return x
