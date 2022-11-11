import torch
from torch import nn
from init_weights import init_weights

__all__ = ['ScoatNet']

def convmxm(in_planes, out_planes, stride=1, groups=1, dilation=2, padding=2, m=3):
    """mxm convolution with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(m-1)//2, groups=groups, bias=True, dilation=(m-1)//2)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            conv1x1(F_g, F_int),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            conv1x1(F_l, F_int),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            conv1x1(F_int, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return psi


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBlock(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(SEBlock, self).__init__()

        self.se = SELayer(in_channels)
        self.conv1 = conv3x3(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.se(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
 
class ScoatNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 blockname='Res', **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        # nb_filter = [64, 128, 256, 512, 1024]
        switch = {
            'VGG': VGGBlock, 'Res': ResBlock, 'SE': SEBlock
        }
        BasicBlock = switch.get(blockname, VGGBlock)
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = BasicBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = BasicBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = BasicBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = BasicBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = BasicBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = BasicBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SEBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SEBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SEBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = BasicBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SEBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SEBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = BasicBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SEBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = BasicBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], nb_filter[0])

        # self.att0_1 = Attention_block(nb_filter[0], nb_filter[1], nb_filter[0])
        self.att1_1 = Attention_block(nb_filter[1], nb_filter[2], nb_filter[1])
        self.att2_1 = Attention_block(nb_filter[2], nb_filter[3], nb_filter[2])
        self.att3_1 = Attention_block(nb_filter[3], nb_filter[4], nb_filter[3])
        # self.att0_2 = Attention_block(nb_filter[0], nb_filter[1], nb_filter[0])
        self.att1_2 = Attention_block(nb_filter[1], nb_filter[2], nb_filter[1])
        self.att2_2 = Attention_block(nb_filter[2], nb_filter[3], nb_filter[2])
        # self.att0_3 = Attention_block(nb_filter[0], nb_filter[1], nb_filter[0])
        self.att1_3 = Attention_block(nb_filter[1], nb_filter[2], nb_filter[1])
        # self.att0_4 = Attention_block(nb_filter[0], nb_filter[1], nb_filter[0])


        if self.deep_supervision:
            self.final1 = conv3x3(nb_filter[0], num_classes)
            self.final2 = conv3x3(nb_filter[0], num_classes)
            self.final3 = conv3x3(nb_filter[0], num_classes)
            self.final4 = conv3x3(nb_filter[0], num_classes)
        else:
            self.final = conv3x3(nb_filter[0], num_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # psi = self.att0_1(x0_0, self.up(x1_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        psi = self.att1_1(x1_0, self.up(x2_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0) + psi * self.up(x2_0)], 1))
        # psi = self.att0_2(x0_1, self.up(x1_1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        psi = self.att2_1(x2_0, self.up(x3_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0) + psi * self.up(x3_0)], 1))
        psi = self.att1_2(x1_1, self.up(x2_1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1) + psi * self.up(x2_1)], 1))
        # psi = self.att0_3(x0_2, self.up(x1_2))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        psi = self.att3_1(x3_0, self.up(x4_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0) + psi * self.up(x4_0)], 1))
        psi = self.att2_2(x2_1, self.up(x3_1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1) + psi * self.up(x3_1)], 1))
        psi = self.att1_3(x1_2, self.up(x2_2))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2) + psi * self.up(x2_2)], 1))
        # psi = self.att0_4(x0_3, self.up(x1_3))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

