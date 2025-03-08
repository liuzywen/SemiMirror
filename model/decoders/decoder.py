import torch
import torch.nn as nn
from torch.nn import functional as F
from model.conpoment.sync_batchnorm import SynchronizedBatchNorm2d
import math
from model.base import  ASPP, get_syncbn

class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class fuse_enhance(nn.Module):
    def __init__(self, infeature, dim=128):
        super(fuse_enhance, self).__init__()
        self.dim = dim
        # self.rgb_conv = nn.Sequential(nn.Conv2d(self.plane[3], self.dim, 1, 1, 0, bias=False), nn.BatchNorm2d(self.dim), nn.ReLU(True))
        # self.d_conv = nn.Sequential(nn.Conv2d(self.plane[3], self.dim, 1, 1, 0, bias=False), nn.BatchNorm2d(self.dim), nn.ReLU(True))

        self.depth_channel_attention = ChannelAttention(infeature)
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

    def forward(self,r,d):
        assert r.shape == d.shape,"rgb and depth should have same size"

        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)

        r_out = r * r_ca
        d_out = d * d_ca
        return r_out, d_out

class Decoder(nn.Module):
    def __init__(self, dim=32, rep_head=True, num_classes = 2):
        super(Decoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.n_class = num_classes
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.plane = [256, 512, 1024, 2048]

        self.Conv3 = nn.Sequential(nn.Conv2d(self.plane[3]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv2 = nn.Sequential(nn.Conv2d(self.plane[2]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv1 = nn.Sequential(nn.Conv2d(self.plane[1]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv0 = nn.Sequential(nn.Conv2d(self.plane[0]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))

        self.Conv64_rgb = nn.Sequential(
                ConvBlock(1, self.dim*4, self.dim*4, normalization='batchnorm'),
                # nn.Dropout2d(0.5),
                nn.Conv2d(self.dim*4, self.n_class, 1, padding=0),
            )

    def forward(self, feature_list, feature_depth):
        R0, R1, R2, R3 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]#1/4 1/4 1/8 1/16 1/32
        D0, D1, D2, D3 = feature_depth[0], feature_depth[1], feature_depth[2], feature_depth[3]
        # out_list = []
        # out_list.append(R3)
        # R = self.up2(R3) ##1/16
        # # print(R0.shape,R1.shape,R2.shape,R3.shape)
        # R = torch.cat((R, R2), dim=1)
        # out = self.Conv32(R)
        # out_list.append(out)
        #
        # R = self.up2(out)
        # R = torch.cat((R, R1),dim=1)
        # out = self.Conv21(R)
        # out_list.append(out)
        #
        # R = self.up2(out)
        # R = torch.cat((R, R0),dim=1)
        # out = self.Conv10(R)
        # out_list.append(out)
        #
        # out = self.up4(out)
        # out = self.Conv64_rgb(out)
        #
        # out_list.append(out)
        # out_list.reverse()
        # out = {'pred': out_list[0]}
        # if self.rep_head:
        #     out["rep"] = out_list[1]
        # return out

        n, _, h, w = R3.shape
        R3 = torch.cat((R3, D3), dim=1)

        R3 = self.Conv3(R3)
        R3 = F.interpolate(R3, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=True)


        R2 = torch.cat((R2, D2), dim=1)
        R2 = self.Conv2(R2)
        R2 = F.interpolate(R2, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=True)


        R1 = torch.cat((R1, D1), dim=1)
        R1 = self.Conv1(R1)
        R1 = F.interpolate(R1, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=True)


        R0 = torch.cat((R0, D0), dim=1)
        R0 = self.Conv0(R0)

        R = torch.cat((R3,R2,R1,R0), dim=1)
        out = self.Conv64_rgb(R)
        out = self.up4(out)

        out = {'pred': out}
        if self.rep_head:
            out["rep"] = R0
        return out




def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                     kernel_size=3, padding=1, stride=stride, bias=False)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class CDecoder(nn.Module):###级联上采样的解码
    def __init__(self, dim=32, rep_head=True, num_classes=2):
        super(CDecoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.n_class = num_classes
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.plane = [256, 512, 1024, 2048]
        # self.plane = [64, 128, 320, 512]
        self.Conv3 = nn.Sequential(nn.Conv2d(self.plane[3] , self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv2 = nn.Sequential(nn.Conv2d(self.plane[2], self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv1 = nn.Sequential(nn.Conv2d(self.plane[1], self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv0 = nn.Sequential(nn.Conv2d(self.plane[0], self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))

        # print(self.dim*4)
        self.Conv64_rgb = nn.Sequential(
            # ConvBlock(1,  self.dim*4, self.dim, normalization='batchnorm'),
            nn.Conv2d(self.dim * 4, self.dim * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim * 4),
            nn.ReLU(True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(self.dim * 4, self.n_class, 1, padding=0),
        )

    def forward(self, feature_list):
        R0, R1, R2, R3 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]  # 1/4 1/4 1/8 1/16 1/32


        n, _, h, w = R3.shape
        R3 = self.Conv3(R3)
        R3_ = F.interpolate(R3, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        # print(R2.shape, R3.shape)
        R2 = self.Conv2(R2)
        R2_ = F.interpolate(R2, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)


        R1 = self.Conv1(R1)
        R1_ = F.interpolate(R1, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        R0 = self.Conv0(R0)
        # R1_ = F.interpolate(R1, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        R = torch.cat((R3_, R2_, R1_, R0), dim=1)
        # print(R.shape)
        out = self.Conv64_rgb(R)
        out = self.up4(out)

        # out = {'pred': out}
        # if self.rep_head:
        #     out["rep"] = R0
        return out



class PVTDecoder(nn.Module):
    def __init__(self, dim=32, rep_head=True, num_classes = 2):
        super(PVTDecoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.n_class = num_classes
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.plane = [64, 128, 320, 512]

        self.Conv3 = nn.Sequential(nn.Conv2d(self.plane[3]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv2 = nn.Sequential(nn.Conv2d(self.plane[2]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv1 = nn.Sequential(nn.Conv2d(self.plane[1]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))
        self.Conv0 = nn.Sequential(nn.Conv2d(self.plane[0]*2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False))

        # self.conv32_2 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),nn.ReLU(True))
        # self.conv21_1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),nn.ReLU(True))
        # self.conv10_0 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),nn.ReLU(True))

        # self.fuse_enhance3 = fuse_enhance(self.plane[3])
        # self.fuse_enhance2 = fuse_enhance(self.plane[2])
        # self.fuse_enhance1 = fuse_enhance(self.plane[1])
        # self.fuse_enhance0 = fuse_enhance(self.plane[0])

        self.Conv64_rgb = nn.Sequential(
                ConvBlock(1, self.dim*4, self.dim, normalization='batchnorm'),
                # nn.Dropout2d(0.1),
                nn.Conv2d(self.dim, self.n_class, 1, padding=0),
            )

    def forward(self, feature_list, feature_depth):
        R0, R1, R2, R3 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]#1/4 1/4 1/8 1/16 1/32
        D0, D1, D2, D3 = feature_depth[0], feature_depth[1], feature_depth[2], feature_depth[3]
        # out_list = []
        # out_list.append(R3)
        # R = self.up2(R3) ##1/16
        # # print(R0.shape,R1.shape,R2.shape,R3.shape)
        # R = torch.cat((R, R2), dim=1)
        # out = self.Conv32(R)
        # out_list.append(out)
        #
        # R = self.up2(out)
        # R = torch.cat((R, R1),dim=1)
        # out = self.Conv21(R)
        # out_list.append(out)
        #
        # R = self.up2(out)
        # R = torch.cat((R, R0),dim=1)
        # out = self.Conv10(R)
        # out_list.append(out)
        #
        # out = self.up4(out)
        # out = self.Conv64_rgb(out)
        #
        # out_list.append(out)
        # out_list.reverse()
        # out = {'pred': out_list[0]}
        # if self.rep_head:
        #     out["rep"] = out_list[1]
        # return out

        n, _, h, w = R3.shape
        # R3_,D3_ = self.fuse_enhance3(R3, D3)
        # R3_, D3_ = R3, D3
        # mul_fa = R3 * D3
        # add_fea = R3 + D3
        # R3_ = mul_fea
        # D3_ = add_fea
        R3 = torch.cat((R3, D3), dim=1)

        R3 = self.Conv3(R3)
        # R3_ = self.linear_c3(R3).permute(0, 2, 1).reshape(n, -1, R3.shape[2], R3.shape[3])
        R3 = F.interpolate(R3, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        # R2_, D2_ = self.fuse_enhance2(R2, D2)
        # R2_, D2_ = R2, D2
        # mul_fa = R2 * D2
        # add_fea = R2 + D2
        R2 = torch.cat((R2, D2), dim=1)
        R2 = self.Conv2(R2)
        # R2_ = self.linear_c2(R2).permute(0, 2, 1).reshape(n, -1, R2.shape[2], R2.shape[3])
        R2 = F.interpolate(R2, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        # R1_, D1_ = self.fuse_enhance1(R1, D1)
        # R1_, D1_ = R1, D1
        # R1 = torch.cat((R1_, D1_), dim=1)
        # mul_fa = R1 * D1
        # add_fea = R1 + D1
        R1 = torch.cat((R1, D1), dim=1)
        R1 = self.Conv1(R1)
        # R1_ = self.linear_c1(R1).permute(0, 2, 1).reshape(n, -1, R1.shape[2], R1.shape[3])
        R1 = F.interpolate(R1, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        # R0_, D0_ = self.fuse_enhance0(R0, D0)
        # R0_, D0_ = R0, D0
        # R0 = torch.cat((R0_, D0_), dim=1)
        # mul_fa = R0 * D0
        # add_fea = R0 + D0
        R0 = torch.cat((R0, D0), dim=1)
        R0 = self.Conv0(R0)


        R = torch.cat((R3,R2,R1,R0), dim=1)
        # R = torch.cat((R3, R0), dim=1)
        # R = R0
        out = self.Conv64_rgb(R)
        out = self.up4(out)

        out = {'pred': out}
        if self.rep_head:
            out["rep"] = R0
        return out

class PVTDecoder_Single(nn.Module):
    def __init__(self, dim=128, rep_head=True, num_classes=2):
        super(PVTDecoder_Single, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.plane = [64, 128, 320, 512]
        self.Conv4 = nn.Sequential(nn.Conv2d(self.plane[2], self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))
        self.Conv3 = nn.Sequential(
                                    nn.Conv2d(self.plane[2], self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))
        self.Conv2 = nn.Sequential(
                                    nn.Conv2d(self.plane[1], self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))

        self.Conv1 = nn.Sequential(
                                     nn.Conv2d(self.plane[0], self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(self.out_dim*4, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, num_classes, 3, 1, 1, bias=False))

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)

    def forward(self, feature_list):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]

        R4 = self.aspp_rgb(R4)
        R4 = self.after_aspp_conv_rgb(R4)
        R4 = F.interpolate(self.Conv4(R4), (R1.shape[2], R1.shape[3]), mode='bilinear', align_corners=False)
        R3 = F.interpolate(self.Conv3(R3), (R1.shape[2], R1.shape[3]), mode='bilinear', align_corners=False)
        R2 = F.interpolate(self.Conv2(R2), (R1.shape[2], R1.shape[3]), mode='bilinear', align_corners=False)
        R1 = F.interpolate(self.Conv1(R1), (R1.shape[2], R1.shape[3]), mode='bilinear', align_corners=False)
        R0 = torch.cat((R4, R3, R2, R1), dim=1)
        sal_map = self.sal_pred(R0)
        sal_out = self.up4(sal_map)
        out = {
            'pred': sal_out, 'R1': R1, 'R2': R2, 'R3': R3, 'R4': R4
        }
        return out

def upsample(in_channels, out_channels, upscale, kernel_size=3):
    # A series of x 2 upsamling until we get to the upscale we want
    layers = []
    conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)
    for i in range(int(math.log(upscale, 2))):
        layers.append(PixelShuffle(out_channels, scale=2))
    return nn.Sequential(*layers)

class DropOutDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.dropout(x)
        return x

class SDecoder(nn.Module): ##depth也出显著图，双解码器
    def __init__(self, dim=32, rep_head=True, num_classes=2):
        super(SDecoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.n_class = num_classes
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.plane = [64, 128, 320, 512]

        self.Conv3 = nn.Sequential(nn.Conv2d(self.plane[2] * 2, self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True))

        self.Conv2 = nn.Sequential(nn.Conv2d(self.plane[2] * 2, self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True))


        self.Conv1 = nn.Sequential(nn.Conv2d(self.plane[1] * 2, self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim,3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True))


        self.Conv0 = nn.Sequential(nn.Conv2d(self.plane[0] * 2, self.dim, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True),
                                   nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False), nn.BatchNorm2d(self.dim),
                                   nn.ReLU(True))


        self.Conv64_rgb = nn.Sequential(
            nn.Conv2d(self.dim*4, self.dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            nn.Conv2d(self.dim, self.n_class, 3, 1, 1, bias=False),
        )
        self.Conv64_noise = nn.Sequential(
            nn.Conv2d(self.dim * 4, self.dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            nn.Conv2d(self.dim, self.n_class, 3, 1, 1, bias=False),
        )

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 320,output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 320,output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)
        self.after_aspp_conv_depth = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)

        # self.line2 = nn.Conv2d(self.dim, self.n_class, kernel_size=3, stride=1, padding=1)
        # self.line3 = nn.Conv2d(self.dim, self.n_class, kernel_size=3, stride=1, padding=1)
        # self.noise0 = DropOutDecoder(1,self.dim*4,self.dim*4)
        # self.noise1 = FeatureDropDecoder(1, 256, 256)
        # self.noise2 = FeatureNoiseDecoder(1, 128, 128)

    def forward(self, feature_list, feature_depth):
        R0, R1, R2, R3 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]  # 1/4 1/4 1/8 1/16 1/32
        D0, D1, D2, D3 = feature_depth[0], feature_depth[1], feature_depth[2], feature_depth[3]


        '''融合特征解码'''
        n, _, h, w = R3.shape
        R3 = self.aspp_rgb(R3)
        D3 = self.aspp_depth(D3)
        R3 = self.after_aspp_conv_rgb(R3)
        D3 = self.after_aspp_conv_depth(D3)
        F3 = torch.cat((R3, D3), dim=1)
        F3 = self.Conv3(F3)
        F3 = F.interpolate(F3, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        F2 = torch.cat((R2, D2), dim=1)
        F2 = self.Conv2(F2)
        F2 = F.interpolate(F2, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)
        # mask2 =  F.interpolate(self.line2(F2),)

        F1 = torch.cat((R1, D1), dim=1)
        F1 = self.Conv1(F1)
        F1 = F.interpolate(F1, (R0.shape[2], R0.shape[3]), mode='bilinear', align_corners=False)

        F0 = torch.cat((R0, D0), dim=1)
        F0 = self.Conv0(F0)

        R = torch.cat((F3, F2, F1, F0), dim=1)
        # if self.training:
        # R_1 = self.noise0(R)
        # out_noise = self.Conv64_noise(R_1)
        # out_noise = self.up4(out_noise)

        out_1 = self.Conv64_rgb(R)
        out_1 = self.up4(out_1)


        out = {'pred': out_1, 'F0':F0, 'F1':F1, 'F2':F2, 'F3':F3}
        if self.rep_head:
            out["rep"] = R0
        return out


class feature_fuse(nn.Module):
    def __init__(self, in_channel=128, out_channel=128):
        super(feature_fuse, self).__init__()
        self.dim = in_channel
        self.out_dim = out_channel
        self.fuseconv = nn.Sequential(nn.Conv2d(2 * self.dim, self.out_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))
        self.conv = nn.Sequential(nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(True))
    def forward(self, Ri, Di):
        assert Ri.ndim == 4
        RDi = torch.cat((Ri, Di), dim=1)
        RDi = self.fuseconv(RDi)
        RDi = self.conv(RDi)
        return RDi


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:],
                                     mode='bilinear', align_corners=True)
        out = img_features

        # edge_features = F.interpolate(edge, x_size[2:],
        #                               mode='bilinear', align_corners=True)
        # edge_features = self.edge_conv(edge_features)
        # out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class HDecoder(nn.Module):
    def __init__(self, dim=128, rep_head=True, num_classes=2):
        super(HDecoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head
        self.fuse1 = feature_fuse(in_channel=64, out_channel=128)
        self.fuse2 = feature_fuse(in_channel=128, out_channel=128)
        self.fuse3 = feature_fuse(in_channel=320, out_channel=128)
        self.fuse4 = feature_fuse(in_channel=320, out_channel=128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))

        self.Conv432 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))
        self.Conv4321 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))

        self.sal_pred = nn.Sequential(nn.Conv2d(self.out_dim, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, num_classes, 3, 1, 1, bias=False))

        self.linear4 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 320,
                                                       output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)
        self.after_aspp_conv_depth = nn.Conv2d(320 * 5, 320, kernel_size=1, bias=False)

        # self.edge_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        # self.rcab_sal_edge = RCAB(32 * 2)
        # self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(True)



    def forward(self, feature_list, feature_list_depth):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        D1, D2, D3, D4 = feature_list_depth[0], feature_list_depth[1], feature_list_depth[2], feature_list_depth[3]

        R4 = self.aspp_rgb(R4)
        D4 = self.aspp_depth(D4)
        R4 = self.after_aspp_conv_rgb(R4)
        D4 = self.after_aspp_conv_depth(D4)

        # print(R1.shape,D1.shape)
        RD1 = self.fuse1(R1, D1)
        RD2 = self.fuse2(R2, D2)
        RD3 = self.fuse3(R3, D3)
        # print(R4.shape,D4.shape)
        RD4 = self.fuse4(R4, D4)

        RD43 = self.up2(RD4)
        RD43 = torch.cat((RD43, RD3), dim=1)
        RD43 = self.Conv43(RD43)

        RD432 = self.up2(RD43)
        RD432 = torch.cat((RD432, RD2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)
        RD4321 = torch.cat((RD4321, RD1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.sal_pred(RD4321)
        sal_out = self.up4(sal_map)

        # mask4 = F.interpolate(self.linear4(RD4), size=sal_out.size()[2:], mode='bilinear', align_corners=False)
        # mask3 = F.interpolate(self.linear3(RD43), size=sal_out.size()[2:], mode='bilinear', align_corners=False)
        # mask2 = F.interpolate(self.linear4(RD432), size=sal_out.size()[2:], mode='bilinear', align_corners=False)
        out = {
            'pred': sal_out,
            # 'mask2':mask2,
            # 'mask3': mask3,
            # 'mask4': mask4,
        }
        if self.rep_head:
            out["rep"] = RD4321
        return out

class SHDecoder(nn.Module):
    def __init__(self, dim=128, rep_head=True, num_classes=2):
        super(SHDecoder, self).__init__()
        self.dim = dim
        self.out_dim = dim
        self.rep_head = rep_head


        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = nn.Sequential(nn.Conv2d(2*self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))
        self.Conv43_1 = nn.Sequential(
                                    nn.Conv2d(1024, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                    nn.BatchNorm2d(self.out_dim),
                                    nn.ReLU(True))
        self.Conv43_2 = nn.Sequential(
            nn.Conv2d(1024, self.out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(True),
            nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(True))

        self.Conv432 = nn.Sequential(
                                     nn.Conv2d(2 * self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))
        self.Conv432_1 = nn.Sequential(
                                     nn.Conv2d(512, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(self.out_dim),
                                     nn.ReLU(True))

        self.Conv4321 = nn.Sequential(nn.Conv2d(2 * self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))
        self.Conv4321_1 = nn.Sequential(nn.Conv2d(256, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True), nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(self.out_dim),
                                      nn.ReLU(True))


        self.sal_pred = nn.Sequential(nn.Conv2d(self.out_dim, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(True),
                                      nn.Conv2d(64, num_classes, 3, 1, 1, bias=False))

        self.aspp_rgb = _AtrousSpatialPyramidPoolingModule(2048, 1024,
                                                       output_stride=16)
        self.after_aspp_conv_rgb = nn.Conv2d(1024 * 5, 1024, kernel_size=1, bias=False)

        self.sal_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(True)



    def forward(self, feature_list):
        R1, R2, R3, R4 = feature_list[0], feature_list[1], feature_list[2], feature_list[3]
        # D1, D2, D3, D4 = feature_list_depth[0], feature_list_depth[1], feature_list_depth[2], feature_list_depth[3]

        R4 = self.aspp_rgb(R4)
        R4 = self.after_aspp_conv_rgb(R4)


        RD43 = self.up2(R4)
        RD43 = self.Conv43_1(RD43)
        R3 = self.Conv43_2(R3)
        RD43 = torch.cat((RD43, R3), dim=1)
        RD43 = self.Conv43(RD43)

        RD432 = self.up2(RD43)
        R2 = self.Conv432_1(R2)
        RD432 = torch.cat((RD432, R2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)
        R1 = self.Conv4321_1(R1)
        RD4321 = torch.cat((RD4321, R1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.sal_pred(RD4321)
        sal_out = self.up4(sal_map)

        return sal_out