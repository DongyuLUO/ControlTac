# Recovered and modified for ControlTac; see THIRD_PARTY_NOTICES.md.
import math
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Attention as Attention_
from timm.models.vision_transformer import Mlp
from .norms import RMSNorm
from .utils import get_same_padding, to_2tuple
from typing import List, Optional, Tuple, Union


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False, **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            # not used for now
            self.q_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = RMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(B, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        # Use PyTorch’s built-in scaled-dot-product attention.
        # Dimensions: q, k, v are (B, N, H, D).
        # We transpose for the standard (B, H, N, D) shape expected by scaled_dot_product_attention.
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)  # (B, H, N, D)
        v = v.transpose(1, 2)  # (B, H, N, D)

        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LiteLA(Attention_):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = RMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        # Optionally compute in FP32 if needed:
        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # v shape: (B, h, h_d, N)
        # pad last row with constant PAD_VAL
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)
        out = out.view(B, C, N).permute(0, 2, 1)
        out = self.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()

class PAGCFGIdentitySelfAttnProcessorLiteLA:
    r"""Self Attention with Perturbed Attention & CFG Guidance"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, image_rotary_emb=None, block_id=None) -> torch.Tensor:
        x_uncond, x_org, x_ptb = x.chunk(3)
        x_org = torch.cat([x_uncond, x_org])
        B, N, C = x_org.shape

        qkv = self.attn.qkv(x_org).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, image_rotary_emb, use_real_unbind_dim=-2)

        out = self.attn.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        # perturbed path (identity attention)
        v_weight = self.attn.qkv.weight[C * 2 : C * 3, :]  # Shape: (dim, dim)
        if self.attn.qkv.bias:
            v_bias = self.attn.qkv.bias[C * 2 : C * 3]  # Shape: (dim,)
            x_ptb = (torch.matmul(x_ptb, v_weight.t()) + v_bias).to(dtype)
        else:
            x_ptb = torch.matmul(x_ptb, v_weight.t()).to(dtype)
        x_ptb = self.attn.proj(x_ptb)

        out = torch.cat([out, x_ptb])

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class PAGIdentitySelfAttnProcessorLiteLA:
    r"""Self Attention with Perturbed Attention Guidance"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, image_rotary_emb=None, block_id=None) -> torch.Tensor:
        x_org, x_ptb = x.chunk(2)
        B, N, C = x_org.shape

        qkv = self.attn.qkv(x_org).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, image_rotary_emb, use_real_unbind_dim=-2)

        out = self.attn.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        # perturbed path (identity attention)
        v_weight = self.attn.qkv.weight[C * 2 : C * 3, :]  # Shape: (dim, dim)
        if self.attn.qkv.bias:
            v_bias = self.attn.qkv.bias[C * 2 : C * 3]  # Shape: (dim,)
            x_ptb = (torch.matmul(x_ptb, v_weight.t()) + v_bias).to(dtype)
        else:
            x_ptb = torch.matmul(x_ptb, v_weight.t()).to(dtype)
        x_ptb = self.attn.proj(x_ptb)

        out = torch.cat([out, x_ptb])

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


class SelfAttnProcessorLiteLA:
    r"""Self Attention with Lite Linear Attention"""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, x: torch.Tensor, mask=None, HW=None, image_rotary_emb=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW
        qkv = self.attn.qkv(x).reshape(B, N, 3, C)
        # B, N, 3, C --> B, N, C
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        q = self.attn.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.attn.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.attn.dim, self.attn.dim, N)  # (B, h, h_d, N)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, image_rotary_emb, use_real_unbind_dim=-2)

        out = self.attn.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.attn.proj(out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            out = out.clip(-65504, 65504)

        return out


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Sana
            cos = cos.transpose(-1, -2)
            sin = sin.transpose(-1, -2)
            x_real, x_imag = x.reshape(*x.shape[:-2], -1, 2, x.shape[-1]).unbind(-2)  # [B, H, D//2, S]
            x_rotated = torch.stack([-x_imag, x_real], dim=-2).flatten(2, 3)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

class FlashAttention(Attention_):
    """Multi-head Flash Attention block with qk norm, but no xformers usage."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, mask=None, HW=None, block_id=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # Build an attention mask, if necessary:
        attn_bias = None
        if mask is not None:
            # (B, 1, N, N) -> repeated across heads
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]],
                                    dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(
                mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf")
            )

        # Fall back to standard scaled_dot_product_attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            x = x.clip(-65504, 65504)

        return x


class Attention(Attention_):
    def forward(self, x, HW=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q, k = q.float(), k.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class T2IFinalLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        kernel_size = kernel_size or patch_size[0]
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0]), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (W == self.img_size[1]), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbedMS(nn.Module):
    """2D Image to Patch Embedding (multi-scale)"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        kernel_size=None,
        padding=0,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        if not padding and kernel_size % 2 > 0:
            padding = get_same_padding(kernel_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x




class ForceEmbedder(nn.Module):
    """
    Embeds a 6D force vector [Fx, Fy, Fz, Tx, Ty, Tz] into a hidden-size vector.
    """
    def __init__(self, hidden_size=1152):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, force):
        """
        force: (B, 3)
        returns: (B, hidden_size)
        """
        return self.mlp(force)

class ContactEmbedder(nn.Module):
    """
    Embeds a 3D TCP_wrt_sensor vector [X, Y, angle] into a hidden-size vector.
    """
    def __init__(self, hidden_size=1152):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, contact):
        """
        contact: (B, 3)
        returns: (B, hidden_size)
        """
        return self.mlp(contact)

class ContactEmbedder_O(nn.Module):
    """
    Embeds a 3D TCP_wrt_sensor vector [X, Y, angle] into a hidden-size vector.
    """
    def __init__(self, hidden_size=1152):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, contact):
        """
        contact: (B, 3)
        returns: (B, hidden_size)
        """
        return self.mlp(contact)
