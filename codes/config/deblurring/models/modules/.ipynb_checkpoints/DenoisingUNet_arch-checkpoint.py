import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv3d,
    ResBlock3D,
    LinearAttention3D, Attention3D,
    PreNorm, Residual)


class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4):
        super().__init__()
        self.depth = depth

        block_class = functools.partial(ResBlock3D, conv=default_conv3d, act=NonLinearity())

         # 초기 컨볼루션 레이어는 입력 채널의 수가 2배가 되도록 변경합니다. (in_nc + cond_nc)
        self.init_conv = default_conv3d(in_nc*2, nf, kernel_size=7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention3D(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv3d(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention3D(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv3d(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention3D(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv3d(nf, out_nc, 3, 1, 1)
    
    def check_image_size(self, x, d, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_d = (s - d % s) % s
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x


    def forward(self, xt, cond, time):
        if x.dim() != 5:
            raise ValueError(f"Input must be a 5D tensor but got {x.dim()}D tensor")        
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time], dtype=torch.float32).to(xt.device)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)
    
        # 3D 데이터의 깊이(D), 높이(H), 너비(W)를 고려
        D, H, W = x.shape[2:]
        x = self.check_image_size(x, D, H, W)
    
        x = self.init_conv(x)
        x_ = x.clone()
    
        t = self.time_mlp(time)
    
        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
    
        # 3D 데이터에 맞게 원본 입력 데이터의 크기로 조정
        x = x[..., :D, :H, :W]
        
        return x

