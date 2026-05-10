# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models import create_model
from einops import rearrange
import numpy as np
import sys
sys.path.append('.')
from .KGFPN import KGFPN, KGFPN_dino
from .dinov2 import DinoV2_self

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 return_mf = False
                 ):
        super().__init__()
        self.return_mf = return_mf
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # feature_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # feature_list.append(x)

        # return self.norm(x.mean([-2, -1])), feature_list[-3:]
        return self.norm(x.mean([-2, -1])), x
    
    def forward_features_score(self, x):
        feature_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_list.append(x)

        return feature_list[-3], feature_list[-2], feature_list[-1]

    def forward(self, x):
        if self.return_mf == False:
            x = self.forward_features(x) 
            return x
        else:
            x = self.forward_features_score(x)
            return x

class ConvNeXt_Score(nn.Module):

    def __init__(self, pretrained = False
                 ):
        super().__init__()
        self.convnext = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
                                 return_mf = True)
        
        # self.norm = nn.LayerNorm(1024)

        self.KGFPN_1 = KGFPN(embed_dim= 512)
        self.KGFPN_2 = KGFPN(embed_dim= 1024)

        self.norm_1 = nn.LayerNorm(256)
        self.norm_2 = nn.LayerNorm(512)
        self.norm_3 = nn.LayerNorm(1024)

        self.downsample_1 = nn.Conv2d(128, 256, kernel_size=8, stride=8)
        self.norm_s1 = LayerNorm(256, eps=1e-6, data_format="channels_first")
        self.downsample_2 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.norm_s2 = LayerNorm(512, eps=1e-6, data_format="channels_first")
      
        if pretrained:
            # path = '/home/lhg/.cache/torch/hub/checkpoints/convnext_base_22k_1k_224.pth'
            path = 'pretrained/convnext_base_22k_1k_224.pth'
            checkpoint = torch.load(path, map_location="cpu")
            self.convnext.load_state_dict(checkpoint["model"], strict=False)
            # self.mask_net.load_state_dict(checkpoint["model"], strict=False)
            print('load convnext_base from:', path)

    def Rearrange(self, x, flag = 0):
        if flag == 0:
            x = rearrange(x,'b c h w -> b (h w) c')
        else:
            hw = x.shape[1]
            H = W = int(np.sqrt(hw))
            x = rearrange(x,'b (h w) c -> b c h w', h = H, w = W)

        return x

    def forward(self, x, x_score = None, wo_KGFPN = False):  
       
        I_q_1, I_q_2, I_q_3 = self.convnext(x)

        # print(x_score.shape)
        # print(I_q_1.shape, I_q_2.shape, I_q_3.shape)
        if wo_KGFPN == False:
            x_score_1 = self.downsample_1(x_score)
            x_score_1 = self.norm_s1(x_score_1)

            x_score_2 = self.downsample_2(x_score_1)
            x_score_2 = self.norm_s2(x_score_2)

            x_score_1, x_score_2 = self.Rearrange(x_score_1, flag = 0), self.Rearrange(x_score_2, flag = 0)
        
            I_q_1, I_q_2, I_q_3 = self.Rearrange(I_q_1, flag = 0), self.Rearrange(I_q_2, flag = 0), self.Rearrange(I_q_3, flag = 0)

            I_q_1, I_q_2, I_q_3 = self.norm_1(I_q_1), self.norm_2(I_q_2), self.norm_3(I_q_3)

            O_q_1 = self.KGFPN_1(I_q_1, x_score_1, I_q_2)

            O_q_2 = self.KGFPN_2(O_q_1, x_score_2, I_q_3)

            O_q_2 = self.Rearrange(O_q_2, flag = 1)

            O_q_final = self.convnext.norm(O_q_2.mean([-2, -1]))
        else:
            # print('use')
            O_q_2  = I_q_3
            O_q_final = self.convnext.norm(O_q_2.mean([-2, -1]))

        # O_q_2 = I_q_3
        # O_q_final = self.convnext.norm(O_q_2.mean([-2, -1]))

        return O_q_final, O_q_2 


class DinoV2(nn.Module):

    def __init__(self, pretrained = False
                 ):
        super().__init__()
        self.convnext = DinoV2_self(model_name = 'dinov2_vitb14', layer1 = 11)
        
        # self.norm = nn.LayerNorm(1024)
        self.KGFPN_1 = KGFPN_dino(embed_dim= 768)
        self.KGFPN_2 = KGFPN_dino(embed_dim= 768)

        self.norm_1 = nn.LayerNorm(768)
        self.norm_2 = nn.LayerNorm(768)
        self.norm_3 = nn.LayerNorm(768)

        self.downsample_1 = nn.Conv2d(128, 768, kernel_size=14, stride=14)
        self.norm_s1 = LayerNorm(768, eps=1e-6, data_format="channels_first")
        self.downsample_2 = nn.Conv2d(768, 768, kernel_size=1, stride=1)
        self.norm_s2 = LayerNorm(768, eps=1e-6, data_format="channels_first")

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
      
        # if pretrained:
        #     path = '/home/lhg/.cache/torch/hub/checkpoints/convnext_base_22k_1k_224.pth'
        #     checkpoint = torch.load(path, map_location="cpu")
        #     self.convnext.load_state_dict(checkpoint["model"], strict=False)
        #     # self.mask_net.load_state_dict(checkpoint["model"], strict=False)
        #     print('load convnext_base from:', path)

    def Rearrange(self, x, flag = 0):
        if flag == 0:
            x = rearrange(x,'b c h w -> b (h w) c')
        else:
            hw = x.shape[1]
            H = W = int(np.sqrt(hw))
            x = rearrange(x,'b (h w) c -> b c h w', h = H, w = W)

        return x

    def forward(self, x, x_score = None, wo_KGFPN = False):  
       
        I_q_1, I_q_2, I_q_3 = self.convnext(x)

        # print(x_score.shape)
        # print(I_q_1.shape, I_q_2.shape, I_q_3.shape)

        if wo_KGFPN == False:
            x_score_1 = self.downsample_1(x_score)
            x_score_1 = self.norm_s1(x_score_1)

            x_score_2 = self.downsample_2(x_score_1)
            x_score_2 = self.norm_s2(x_score_2)

            x_score_1, x_score_2 = self.Rearrange(x_score_1, flag = 0), self.Rearrange(x_score_2, flag = 0)
        
            # I_q_1, I_q_2, I_q_3 = self.Rearrange(I_q_1, flag = 0), self.Rearrange(I_q_2, flag = 0), self.Rearrange(I_q_3, flag = 0)

            I_q_1, I_q_2, I_q_3 = self.norm_1(I_q_1), self.norm_2(I_q_2), self.norm_3(I_q_3)

            O_q_1 = self.KGFPN_1(I_q_1, x_score_1, I_q_2)

            O_q_2 = self.KGFPN_2(O_q_1, x_score_2, I_q_3)

            O_q_2 = self.Rearrange(O_q_2, flag = 1)

            O_q_2 = self.pool(O_q_2)

            # print('O_q_2', O_q_2.shape)

            O_q_final = self.convnext.dino_model.norm(O_q_2.mean([-2, -1]))
        else:
            # print('use')
            O_q_2  = I_q_3

            O_q_2 = self.Rearrange(O_q_2, flag = 1)

            O_q_2 = self.pool(O_q_2)

            O_q_final = self.convnext.dino_model.norm(O_q_2.mean([-2, -1]))

        # O_q_2 = I_q_3
        # O_q_final = self.convnext.norm(O_q_2.mean([-2, -1]))

        return O_q_final, O_q_2 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224.pth",
}


@register_model
def convnext_tiny(pretrained=True, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    if pretrained:
        url = model_urls["convnext_tiny_22k"] if in_22k else model_urls["convnext_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        print(url)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_small(pretrained=True, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        print(url)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_base(pretrained=True, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    if pretrained:
        path = '/home/lhg/.cache/torch/hub/checkpoints/convnext_base_22k_1k_224.pth'
        # path = '/home/lhg/work/ssd_new/DRL/DRL_ViT/pretrain_model/convnext_base_22k_224.pth'
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print('load from:', path)
    return model


@register_model
def convnext_large(pretrained=True, in_22k=True, **kwargs):
    # model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        print(url)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def convnext_xlarge(pretrained=True, in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        print(url)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_base_w_score(pretrained=True, in_22k=True, **kwargs):
    # depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
    model = ConvNeXt_Score(pretrained=pretrained)
    return model

@register_model
def dinov2(pretrained=True, in_22k=True, **kwargs):
    # depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
    model = DinoV2(pretrained=pretrained)
    return model

if __name__ == '__main__':
    model = create_model('convnext_tiny', pretrained=True, num_classes=1000)
    print(model)
