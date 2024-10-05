from xml.etree.ElementPath import xpath_tokenizer

import numpy as np
from itertools import repeat
import collections.abc
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from addict import Dict #from attrdict import AttrDict
from utils.misc import stack, logmeanexp
from utils.sampling import sample_subset
from torch.distributions import Normal


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=np.float32)
    grid_w = torch.arange(grid_size, dtype=np.float32)
    grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
    grid = torch.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    # emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim , grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim , grid[1])  # (H*W, D/2)

    emb = torch.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    poshape = pos.shape
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float).to(pos.device) 
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb.reshape(*poshape, -1)

def get_sincos_pos_embed_from_grid(embed_dim, pos):
    if pos.shape[-1] == 2:  # 2D input
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos[:, :, 0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos[:, :, 1])
        emb = torch.cat([emb_h, emb_w], dim=-1)
    else:  # 1D input
        emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos.squeeze(-1))
    return emb

class CrossAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ctx=None) -> torch.Tensor:
        if ctx is None:
          ctx = x  # becomes self-attention
        B, N, C = x.shape
        qkv = self.kv(ctx).reshape(B, ctx.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = qkv.unbind(0)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimeStepEmbedder(nn.Module):
    """
    Embeds scalar values into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class NP_SelfAttention_Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=2.0, **block_kwargs):
        super().__init__()

        # tmp for qkv
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class NP_CrossAttention_Block(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=2.0, **block_kwargs):
        super().__init__()

        # tmp for qkv
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # ====== context block part ===============
        self.ctx_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.ctx_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.ctx_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ctx_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.ctx_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, x, ctx, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa2, scale_msa2, gate_msa2, shift_mlp2, scale_mlp2, gate_mlp2 = self.ctx_adaLN_modulation(c).chunk(6,
                                                                                                                  dim=1)
        #x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        #ctx = ctx + gate_msa2.unsqueeze(1) * self.ctx_attn(modulate(self.ctx_norm1(ctx), shift_msa2, scale_msa2))
        #ctx = ctx + gate_mlp2.unsqueeze(1) * self.ctx_mlp(modulate(self.ctx_norm2(ctx), shift_mlp2, scale_mlp2))

        mod_ctx = ctx  # modulate(self.ctx_norm1(ctx), shift_msa2, scale_msa2)
        B, N, C = mod_ctx.shape
        #x = x + gate_mlp.unsqueeze(1) * self.cross_attn(x, mod_ctx)
        x = x + gate_mlp.unsqueeze(1) * self.cross_attn(modulate(self.norm1(x), shift_msa, scale_msa), 
                                                        modulate(self.ctx_norm1(ctx), shift_msa2, scale_msa2))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class NP_Transformer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            data_size=1,
            posenc_size=20,
            hidden_size=128,
            depth=3,
            num_heads=4,
            mlp_ratio=2.0,
            label_conditioning=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.x_embedder = nn.Linear(data_size, hidden_size, bias=True)
        self.pc_embedder = nn.Linear(posenc_size, hidden_size, bias=True)
        self.vc_embedder = nn.Linear(data_size, hidden_size, bias=True)
        self.pt_embedder = nn.Linear(posenc_size, hidden_size, bias=True)
        self.t_embedder = TimeStepEmbedder(hidden_size)

        self.ctx_blocks = nn.ModuleList([
            NP_SelfAttention_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.in_blocks = nn.ModuleList([
            NP_SelfAttention_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.cross_blocks = nn.ModuleList([
            NP_CrossAttention_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.out_blocks = nn.ModuleList([
            NP_SelfAttention_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, data_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.ctx_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.in_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.cross_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.out_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, pc, vc, pt, t):
        """
        """
        # print(x)
        # print(self.x_embedder)

        # print("x", x.shape)
        # print("pc", pc.shape)
        # print("vc",vc.shape)
        # print("pt", pt.shape)

        x = self.x_embedder(x)  # (N, T, D)
        pc = self.pc_embedder(pc)  # (N, T, D)
        vc = self.vc_embedder(vc)  # (N, T, D)
        pt = self.pt_embedder(pt)  # (N, T, D)
        t = self.t_embedder(t)  # (N, T, D)



        x = x + pt
        ctx = vc + pc

        c = t  # (N, D)
        for block in self.ctx_blocks:
            ctx = block(ctx, c)  # (N, T, D)
        for block in self.in_blocks:
            x = block(x, c)  # (N, T, D)
        for block in self.cross_blocks:
            x = block(x, ctx, c)  # (N, T, D)
        for block in self.out_blocks:
            x = block(x, c)  # (N, T, D)
            
        x = self.final_layer(x, c)  # (N, T,)
        return x


class DNP(nn.Module):
    def __init__(self, 
                 data_dim=1, 
                 query_dim=1, 
                 posenc_dim=40,
                 ouput_scale=0.1, 
                 depth=3, 
                 hidden_size=256,
                 timesteps=1000,
                 clipped_reverse_diffusion=True):
        
        super().__init__()
        self.timesteps=timesteps
        self.data_dim = data_dim
        self.output_scale = ouput_scale
        self.query_dim = query_dim
        self.posenc_dim = posenc_dim
        self.clipped_reverse_diffusion = clipped_reverse_diffusion
        
        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model = NP_Transformer(data_size=self.data_dim, posenc_size=self.posenc_dim,
                                    depth=depth, hidden_size=hidden_size)

    def pred_noise(self, x, pt, vc, pc, noise):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device)
        x_t=self._forward_diffusion(x,t,noise)
        pred_noise=self.model(x=x_t, pt=pt, vc=vc, pc=pc,t=t)

        return pred_noise

    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*torch.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1)*x_0+ \
                self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t, pt, vc, pc, t, noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x=x_t, pt=pt, vc=vc, pc=pc,t=t)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, pt, vc, pc, t, noise):
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x=x_t, pt=pt, vc=vc, pc=pc, t=t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1)

        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise

    
    @torch.no_grad()
    def predict(self, xc, yc, xt, num_samples=None):
        if num_samples is None:
            num_samples = 1
        n_tokens = xt.shape[1]
        # xc = get_2d_sincos_pos_embed_from_grid(self.posenc_dim, xc.squeeze(-1))
        # xt = get_2d_sincos_pos_embed_from_grid(self.posenc_dim, xt.squeeze(-1))

        xc = get_sincos_pos_embed_from_grid(self.posenc_dim, xc)
        xt = get_sincos_pos_embed_from_grid(self.posenc_dim, xt)

        xc = stack(xc, num_samples).reshape(-1, *xc.shape[1:])            
        yc = stack(yc, num_samples).reshape(-1, *yc.shape[1:])
        xt = stack(xt, num_samples).reshape(-1, *xt.shape[1:])
        x_t=torch.randn((xt.shape[0], n_tokens, self.data_dim)).to(xt.device)
        for i in range(self.timesteps-1,-1,-1):
            noise=torch.randn_like(x_t).to(xt.device)
            t=torch.tensor([i for _ in range(xt.shape[0])]).to(xt.device)

            if self.clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t=x_t, pt=xt, vc=yc, pc=xc, t=t ,noise=noise)
            else:
                x_t=self._reverse_diffusion(x_t=x_t, pt=xt, vc=yc, pc=xc, t=t ,noise=noise)

        x_t = x_t.reshape(num_samples, -1, n_tokens, self.data_dim)
        return Normal(x_t, self.output_scale)

    
    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = Dict()
        if self.training:
            noise = torch.randn_like(batch.y).to(batch.y.device)

            xc = get_sincos_pos_embed_from_grid(self.posenc_dim, batch.xc)
            x = get_sincos_pos_embed_from_grid(self.posenc_dim, batch.x)

            pred=self.pred_noise(x=batch.y, pt=x, vc=batch.yc, pc=xc, noise=noise)
            outs.loss=nn.MSELoss()(pred,noise)
      
        else:
            # print("predicting in forward")
            if num_samples is None:
                num_samples = 1
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            y = torch.stack([batch.y]*num_samples)
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                #ll = logmeanexp(py.log_prob(y).sum(-1))
                py_agg = Normal(py.mean.mean(0), py.mean.std(0))
                ll = py_agg.log_prob(y).sum(-1)
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
            else:
                ll = py.log_prob(y).sum(-1)
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]

            # for s in torch.arange(0.05, 0.01, 0.2):
            #     py.scale = s
            #     ll = logmeanexp(py.log_prob(y).sum(-1))
            #     outs['ctx_ll_{}'.format(s)] = ll[...,:num_ctx].mean()
            #     outs['tar_ll_{}'.format(s)] = ll[...,num_ctx:].mean()

        return outs
