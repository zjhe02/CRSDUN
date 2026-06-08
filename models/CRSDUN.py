import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.norm = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        else:
            self.norm1 = nn.GroupNorm(dim, dim)
            self.norm2 = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2, *args, **kwargs):
        if self.norm_type == 'ln':
            x1 = self.norm1(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x2 = self.norm2(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x1 = self.norm1(x1)
            x2 = self.norm2(x2)
        return self.fn(x1, x2, *args, **kwargs)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):

        out = self.net(x)
        return out


class Para_Estimator(nn.Module):
    def __init__(self, in_nc, out_nc=1, channel=32):
        super(Para_Estimator, self).__init__()
        self.fusion = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.,1.,0.01, 0]).view(out_nc,1,1))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=False),
                )
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):

        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x


#----------Core Novelty: CASTAB----------
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CASTA(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        if refine:
            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )
        
    def stoken_forward(self, x, fea):
        '''
           x: (B, C, H, W)
           fea: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h = self.stoken_size
        w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        # 976
        
        stoken_features = F.adaptive_avg_pool2d(fea, (hh, ww)) # (B, C, hh, ww)
        # 955
        
        # 935
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C) # B nh*nw, hw, C
        # 911
        
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, hh*ww, h*w, 9)
                # 874
                affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
                # 871
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                    # 777
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
                    # 853
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
                    # 777            
                    
                    # 771
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, hh, ww)
                    # 767
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, hh*ww, C, 9)
        # 853
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, hh, ww)).reshape(B, C, hh, ww)            
        
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, hh, ww)
        # 767
        
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.reshape(B, C, hh*ww).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, hh, ww)
            else:
                stoken_features = self.stoken_refine(stoken_features)
            
        # 727
        
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9) # (B, hh*ww, C, 9)
        # 714
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, hh*ww, C, h*w)
        # 687
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        
        # 681
        # 591 for 2 iters
                
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        return pixel_features
    
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x, fea):
        if self.stoken_size>1:
            return self.stoken_forward(x, fea)
        else:
            return self.direct_forward(x)


class CASTAB(nn.Module):
    def __init__(self,
            dim,
            dim_head,
            heads,
            stoken_size,
            num_blocks,
            norm_type="bn") -> None:
        super().__init__()
        
        self.pos = nn.Conv2d(dim*2, dim*2, 5, 1, 2, groups=dim*2)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm2(dim, CASTA(dim=dim, stoken_size=stoken_size, num_heads=heads), norm_type="ln"),
                PreNorm(dim, FFN(dim=dim), norm_type=norm_type)
            ]))

    def forward(self, x, fea, switch=False):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = torch.cat([x, fea], dim=1)
        x= self.pos(x) + x
        x, fea = torch.chunk(x, chunks=2, dim=1)
        for i, (attn, ff) in enumerate(self.blocks):
            if switch is True:
                x = attn(fea, x) + x
            else:
                x = attn(x, fea) + x
            x = ff(x) + x

        return x


#----------Rec Module----------
class Mask_embedding(nn.Module):
    def __init__(self, ch=31):
        super().__init__()
        self.cnn = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.mask = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)

    def forward(self, x, mask):
        out = self.cnn(x)*(1+self.mask(mask))
        return out


class WSSA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=31, heads=1, shift=False):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        h1, h2 = h // self.window_size[0], w // self.window_size[1]
        q, k, v = map(lambda t: rearrange(t, 'b c (h1 h) (h2 w) ->b (h1 h2) c (h w)', h1=h1, h2=h2), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b (h1 h2) c (h w) -> b c (h1 h) (h2 w)', h1=h1, h=h // h1)
        out = self.to_out(out)
        return out

    def forward(self, x):

        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0]//2, dims=2).roll(shifts=w_size[1]//2, dims=3)
        out = self.cal_attention(x)
        if self.shift:
            out = out.roll(shifts=-1*w_size[1]//2, dims=3).roll(shifts=-1*w_size[0]//2, dims=2)
        return out


class ERB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=31, heads=1):
        super().__init__()
        self.WSSA = PreNorm(dim, WSSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads,
                                      shift=False))
        self.FFN = PreNorm(dim, FFN(dim=dim), norm_type='gn')

    def forward(self, x):

        x = self.WSSA(x) + x
        x = self.FFN(x) + x
        return x


class CMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 11, 1, 5, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a*v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out


class SAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
            nn.GELU(),
        )
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=.02)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out)
        return out


class ARB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CMB = PreNorm(dim, CMB(dim=dim))
        self.SAB = PreNorm(dim, SAB(dim=dim), norm_type='gn')

    def forward(self, x):
        x = self.CMB(x) + x
        x = self.SAB(x) + x
        return x


class SSRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=31, heads=1):
        super().__init__()

        self.pos = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.SARB = ERB(dim, window_size, dim_head, heads)
        self.SRB = ARB(dim)

    def forward(self, x):

        x = self.pos(x) + x
        x = self.SARB(x)
        x = self.SRB(x)

        return x


#----------Seg Module----------
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W)  # B H' W' C
        x = x.permute(0, 3, 1, 2)
        return x


class SwinBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            shift_size=4
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, WindowAttention(dim, window_size=8, num_heads=heads)),
                PreNorm(dim, FFN(dim=dim))
            ]))
        self.shift_size=shift_size

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # cyclic shift

        for i, (attn, ff) in enumerate(self.blocks):
            if self.shift_size > 0 and i % 2 == 1:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            else:
                x = x
            x = attn(x) + x
            x = ff(x) + x

            # reverse cyclic shift
            if self.shift_size > 0 and i % 2 == 1:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            else:
                x = x

        return x


#----------CAT----------
class CAT_Rec(nn.Module):
    def __init__(self, dim, n_class, cross=True):
        super(CAT_Rec, self).__init__()

        self.cross = cross
        
        self.mask_embedding = Mask_embedding(dim)
        self.down1 = SSRB(dim=dim, dim_head=dim, heads=1)
        self.downsample1 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)
        self.down2 = SSRB(dim=dim*2, dim_head=dim, heads=2)
        self.downsample2 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False)
        self.bottleneck = SSRB(dim=dim*4, dim_head=dim, heads=4)
        
        if cross is True:
            self.up2 = CASTAB(dim=dim*2, dim_head=dim, heads=2, stoken_size=4, num_blocks=1, norm_type="bn")
            self.up1 = CASTAB(dim=dim, dim_head=dim, heads=1, stoken_size=8, num_blocks=1, norm_type="bn")
            self.theta_emb1 = nn.Conv2d(n_class, dim, 1)
            self.theta_emb2 = nn.Sequential(
                nn.Conv2d(n_class, dim, 1),
                nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)
            )
            self.arb2 = ARB(dim=dim*2)
            self.arb1 = ARB(dim=dim)
        else:
            self.up2 = SSRB(dim=dim*2, dim_head=dim, heads=2)
            self.up1 = SSRB(dim=dim, dim_head=dim, heads=1)
        
        self.upsample2 = nn.ConvTranspose2d(dim*4, dim*2, 2, 2)
        self.fusion2 = nn.Conv2d(dim*4, dim*2, 1, 1, 0, bias=False)
        self.upsample1 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.fusion1 = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=False)
        
        self.out = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)

    def forward(self, x, mask, theta=None):

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_in = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        mask = F.pad(mask, [0, pad_w, 0, pad_h], mode='reflect')

        x = self.mask_embedding(x_in, mask)
        x1 = self.down1(x)
        x = self.downsample1(x1)
        x2 = self.down2(x)
        x = self.downsample2(x2)
        x = self.bottleneck(x)
        x = self.upsample2(x)
        x = self.fusion2(torch.cat([x, x2], dim=1))
        if self.cross is True:
            x = self.up2(x, self.theta_emb2(theta))
            x = self.arb2(x)
        else:
            x = self.up2(x)
        x = self.upsample1(x)
        x = self.fusion1(torch.cat([x, x1], dim=1))
        if self.cross is True:    
            x = self.up1(x, self.theta_emb1(theta))
            x = self.arb1(x)
        else:
            x = self.up1(x)
        out = self.out(x) + x_in

        return out[:, :, :h_inp, :w_inp]
   
        
class CAT_Seg(nn.Module):
    def __init__(self, in_dim, dim, out_dim, cross=True):
        super(CAT_Seg, self).__init__()
        self.dim = dim
        self.stage = 2
        self.cross = cross

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(self.stage):
            self.encoder_layers.append(nn.ModuleList([
                SwinBlock(
                    dim=dim_stage, num_blocks=2, dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SwinBlock(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=2)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                CASTAB(
                    dim=dim_stage // 2, num_blocks=2, dim_head=dim,
                    heads=(dim_stage // 2) // dim, stoken_size=8//((dim_stage // 2) // dim), norm_type="bn") if cross is True else SwinBlock(
                    dim=dim_stage // 2, num_blocks=2, dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        if cross is True:
            self.x_emb1 = nn.Conv2d(dim, dim, 1)
            self.x_emb2 = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)
            )

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, kexi, emb=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(kexi)

        if self.cross is True:
            x_emb = []
            x_emb.append(self.x_emb1(emb))
            x_emb.append(self.x_emb2(emb))

        # Encoder
        fea_encoder = []
        for (Block, FeaDownSample) in self.encoder_layers:
            fea = Block(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, fusion, Block) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            if self.cross is True:
                fea = Block(fea, x_emb[self.stage-1-i], switch=True)
            else:
                fea = Block(fea)

        # Mapping
        out = self.mapping(fea)

        return out


#----------CRSDUN----------
class CRSDUN(torch.nn.Module):
    def __init__(self, stage, bands, n_class):
        super(CRSDUN, self).__init__()
        netlayer = []
        self.stage = stage
        self.nC = bands # Spectral Bands
        self.n_class = n_class # Segmentation Classes
        self.phi = nn.Conv2d(n_class, bands, 1, bias=False) # learnable spectral dictionary
        self.initial = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        
        #----------LISTA Params----------
        self.s = nn.Conv2d(n_class, n_class, 1, bias=False)
        self.w = nn.Conv2d(bands, n_class, 1, bias=False)
        
        #----------Degradation Estimation----------
        para_estimator = []
        for i in range(self.stage):
            para_estimator.append(Para_Estimator(in_nc=self.nC+self.n_class, out_nc=4))

        #----------Network----------
        for i in range(self.stage):
            # using self attention only in the first stage for convergence stability
            if i == 0:
                netlayer.append(CAT_Rec(dim=bands, n_class=n_class, cross=False))
                netlayer.append(ARB(dim=bands))
                netlayer.append(CAT_Seg(in_dim=n_class, dim=bands, out_dim=n_class, cross=False))
            else:
                netlayer.append(CAT_Rec(dim=bands, n_class=n_class, cross=True))
                netlayer.append(ARB(dim=bands))
                netlayer.append(CAT_Seg(in_dim=n_class, dim=bands,out_dim=n_class, cross=True))

        self.para_est = nn.ModuleList(para_estimator)
        self.net_stage = nn.ModuleList(netlayer)

    def shift_back(self, x, len_shift=2):
        b, c, h, w = x.shape
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :w-(c-1)*len_shift]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC*2-2, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1).to(g.device)
        PhiTg = temp_1 * Phi_shift
        PhiTg = self.shift_back(PhiTg)
        return PhiTg

    def mul_Phif(self, Phi_shift, z):
        z_shift = self.shift(z)
        Phiz = Phi_shift * z_shift
        Phiz = torch.sum(Phiz, 1)
        return Phiz.unsqueeze(1)
    
    def soft_thre(self, x, b):
        # l_1 norm based
        # implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)
        x = torch.sign(x) * torch.clamp(torch.abs(x) - b, 0)
        return x

    def forward(self, y, input_mask=None):
        Phi, Phi_shift, PhiPhiT = input_mask
        y = y.unsqueeze(1)
        y_normal = y / self.nC*2
        temp_y = y_normal.repeat(1, self.nC, 1, 1)
        x0 = self.shift_back(temp_y)
        x = self.initial(torch.cat([x0, Phi], dim=1))
        B, C, H, W = x.shape
        theta = torch.zeros([B, self.n_class, H, W], device=x.device)
        kexi = torch.zeros([B, self.n_class, H, W], device=x.device)

        x_list = []
        theta_list = []

        for i in range(self.stage):

            #----------Degradation Estimation----------
            paras = self.para_est[i](torch.cat([x, theta], dim=1))
            a, mu, b, gamma = paras[:, 0:1, :, :], paras[:, 1:2, :, :], paras[:, 2:3, :, :], paras[:, 3:4, :, :]
            
            ##----------Reconstruction Sub-problem----------
            phi_theta = self.phi(theta)
            x_new = mu * x + (1 - mu) * phi_theta
            A_x_phitheta = self.mul_Phif(Phi_shift, x_new)
            z = x_new + self.mul_PhiTg(Phi_shift, torch.div(y - A_x_phitheta, PhiPhiT + a))
            
            #----------CAT----------
            if i == 0:
                # self attention
                x = self.net_stage[3 * i](z, Phi)
            else:
                # cross attention
                x = self.net_stage[3 * i](z, Phi, theta)
            #----------Refinement----------
            x = self.net_stage[3 * i + 1](x)
            
            ##----------Segmentation Sub-problem----------
            kexi = self.soft_thre(self.s(kexi) + self.w(x) + gamma * theta, b)
            
            #----------CAT----------
            if i == 0:
                # self attention
                theta = self.net_stage[3 * i + 2](kexi)
                theta = theta + kexi
            else:
                # cross attention
                theta = self.net_stage[3 * i + 2](kexi, x)
                theta = theta + kexi

            x_list.append(x)
            theta_list.append(theta)

        return x_list, theta_list
    

#----------Test Params and FLOPs----------
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table
    ch = 28
    model = CRSDUN(stage=5, bands=ch, n_class=23)
    y = torch.randn(1, 256, 256+(ch-1)*2)
    Phi_shift_back = torch.randn(1, ch, 256, 256)
    Phi = torch.randn(1, ch, 256, 256+(ch-1)*2)
    PhiPhi_T = torch.randn(1, 1, 256, 256+(ch-1)*2)

    flops = FlopCountAnalysis(model, (y, (Phi_shift_back, Phi, PhiPhi_T)))
    print(flops.total() / 1e9)
    print(parameter_count_table(model))
    print(flop_count_table(flops))
