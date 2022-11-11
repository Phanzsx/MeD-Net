from matplotlib.animation import ImageMagickWriter
import torch
from torch import nn, einsum
import torch.nn.functional as F
import os
import numpy as np
import mresnet as resnet
import mdensenet as densenet
import minception as inception
import math

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #defalt element-wise affine: true
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout=0., dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, groups=dim_head)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, groups=1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attn_drop = nn.Dropout(attn_dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, d, n, h = *x.shape, self.heads # b n d
        x = rearrange(x, 'b n d -> b d n')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (d h) n -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.att = PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout, attn_dropout=0.5))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        _, n, _ = x.size()
        # x = rearrange(x, 'b (h w) d -> b d h w', h=int(n**0.5)) # b n d
        # x = self.conv(x)
        # x = rearrange(x, 'b d h w -> b (h w) d') # b n d
        x = self.att(x) + x
        x = self.ff(x) + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, input_channel=3, embed_dim=None):
        super().__init__()

        self.conv = nn.Sequential(
            conv3x3(input_channel, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1), 
        )

    def forward(self, x):
        b, p, c, h, w = x.size()
        x = rearrange(x, 'b p c h w -> (b p) c h w')
        x = self.conv(x)
        # x = rearrange(x, '(b p) c h w -> b p c h w', p=p)
        return x


class Pre_model(nn.Module):
    def __init__(self, *, num_patch, dim=512, num_classes, pool='mean'):
        super().__init__()
        embed_dim = 128
        head_num = 8
        self.patch_embed = PatchEmbed(input_channel=3, embed_dim=embed_dim)
        self.proj = nn.Conv2d(embed_dim*num_patch, dim, kernel_size=1, stride=1)
        self.conv = Bottle2neck(dim, dim)
        self.transformer1 = Transformer(
            dim=dim, heads=head_num, dim_head=dim//head_num, mlp_dim=512, dropout=0.5)
        self.transformer2 = Transformer(
            dim=dim, heads=head_num, dim_head=dim//head_num, mlp_dim=512, dropout=0.5)
        self.transformer3 = Transformer(
            dim=dim, heads=head_num, dim_head=dim//head_num, mlp_dim=512, dropout=0.5)
        self.transformer4 = Transformer(
            dim=dim, heads=head_num, dim_head=dim//head_num, mlp_dim=512, dropout=0.5)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.fc = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        b, p, c, h, w = x.size()
        x = self.patch_embed(x) # (b p) c h w

        x = rearrange(x, '(b p) c h w -> b (p c) h w', b=b)
        x = self.proj(x) # b d h w
        x = self.conv(x)

        x = rearrange(x, 'b d h w -> b (h w) d') # b n d

        x = self.transformer1(x) # b n d
        _, n, _ = x.size()
        x = rearrange(x, 'b (h w) d -> b h w d', h=int(n**0.5))
        x = reduce(x, 'b (h1 h2) (w1 w2) d -> b (h1 w1) d', 'max', h2=2, w2=2)

        x = self.transformer2(x) # b n d
        
        # _, n, _ = x.size()
        # x = rearrange(x, 'b (h w) d -> b h w d', h=int(n**0.5))
        # x = reduce(x, 'b (h1 h2) (w1 w2) d -> b (h1 w1) d', 'max', h2=2, w2=2)

        x = self.transformer3(x) # b n d
        x = self.transformer4(x) # b n d

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        out = self.mlp_head(x)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    num_patch = 512
    image = torch.rand(1, num_patch, 3, 28, 28).cuda(0)
    image2 = torch.rand(1, 3, 512, 512).cuda(0)
    pos0 = (torch.rand(1, num_patch)*8).long().cuda(0)
    pos1 = (torch.rand(1, num_patch)*8).long().cuda(0)
    pos2 = (torch.rand(1, num_patch)*50).long().cuda(0)
    model = Pre_model(num_patch=num_patch, dim=512, \
        num_classes=3, pool='mean')
    model = model.cuda(0)
    out = model(image)
    print(out.size())