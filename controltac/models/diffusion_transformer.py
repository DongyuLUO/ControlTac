# Recovered and modified for ControlTac; see THIRD_PARTY_NOTICES.md.
import os
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .modules import DWMlp, GLUMBConv, MBConvPreGLU, Mlp
from .blocks import (
    Attention,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbed,
    TimestepEmbedder,
    T2IFinalLayer,
    ForceEmbedder,
    t2i_modulate,
)
from .norms import RMSNorm



####################################################################################################
#  FTBlock: A Transformer block with global shared adaptive layer norm (adaLN-Zero) conditioning   #
####################################################################################################
class FTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type="flash",
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        # Layer normalization (without learnable affine parameters)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Choose self-attention type
        if attn_type == "flash":
            # Flash self-attention
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif attn_type == "linear":
            # Linear self-attention; here we compute the number of heads from head dimension
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "vanilla":
            # Vanilla self-attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        # Multi-head cross-attention layer
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Choose feed-forward network type
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu,
                drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=approx_gelu,
                drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # Learnable parameters for adaptive modulation (scale and shift)
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, **kwargs):
        B, N, C = x.shape
        # Split adaptive modulation parameters for self-attention and MLP branches
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        # Apply self-attention with adaptive modulation
        x = x + self.drop_path(
            gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C)
        )
        # Apply cross-attention
        x = x + self.cross_attn(x, y)
        # Apply feed-forward network with adaptive modulation
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


####################################################################################################
#      Tactile diffusion model with a Transformer backbone supporting non-square input sizes           #
####################################################################################################
class TactileDiffusionTransformer(nn.Module):
    def __init__(
        self,
        input_size=(240, 320),
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        pred_sigma=True,
        drop_path: float = 0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=120,
        qk_norm=False,
        force_norm=False,
        norm_eps=1e-5,
        attn_type="flash",
        ffn_type="mlp",
        use_pe=True,
        force_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels//2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.depth = depth
        self.use_pe = use_pe
        self.force_norm = force_norm
        self.fp32_attention = kwargs.get("use_fp32_attention", False)

        kernel_size = patch_embed_kernel or patch_size
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.force_embedder = ForceEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches

        # Define base_size as the number of patches (height, width)
        self.base_size = (input_size[0] // patch_size, input_size[1] // patch_size)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if self.force_norm:
            self.attention_force_norm = RMSNorm(hidden_size, scale_factor=force_norm_scale_factor, eps=norm_eps)

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                FTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size = (input_size[0] // patch_size, input_size[1] // patch_size),
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def forward(self, x, timestep, force, data_info=None, **kwargs):
        """
        Args:
            x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
            timestep: (N,) tensor of diffusion timesteps
            force: (N, 1, L, D) tensor of force inputs
        Returns:
            x: Reconstructed tensor of shape (N, out_channels, H, W)
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        force = force.to(self.dtype)
        # Ensure positional embeddings are on the same dtype
        pos_embed = self.pos_embed.to(self.dtype)
        # Calculate patch grid dimensions (height and width)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if self.use_pe:
            x = self.x_embedder(x) + pos_embed  # (N, T, D)
        else:
            x = self.x_embedder(x)
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        force = self.force_embedder(force)
        if self.force_norm:
            force = self.attention_force_norm(force)
        for block in self.blocks:
            x = block(x, force, t0)  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, patch_size**2 * out_channels)
        x = self.unpatchify(x)      # (N, out_channels, H, W)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(self, x, timestep, y, **kwargs):
        """
        For DPM-Solver, variance prediction is not needed.
        """
        model_out = self.forward(x, timestep, y)
        return model_out.chunk(2, dim=1)[0] if self.pred_sigma else model_out

    def unpatchify(self, x):
        """
        Rearranges the patches back into the image.
        Args:
            x: (N, T, patch_size**2 * C)
        Returns:
            imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        # Use precomputed patch grid dimensions
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        # Rearrange dimensions to get the final image shape (N, C, H*p, W*p)
        x = torch.einsum("n h w p q c -> n c h p w q", x)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def initialize_weights(self):
        # Initialize transformer layers using Xavier uniform initialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.use_pe:
            # Generate positional embeddings using the provided positional encoding function
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.base_size[0],
                self.base_size[1],
                cls_token=False,
                extra_tokens=0,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize the patch embedding layer (similar to a linear layer)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP (assuming self.y_embedder is defined)
        nn.init.normal_(self.force_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.force_embedder.mlp[2].weight, std=0.02)



    @property
    def dtype(self):
        return next(self.parameters()).dtype


####################################################################################################
#    Positional Encoding Functions: Generate 2D sine-cosine embeddings for patch positions        #
####################################################################################################
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w=None, cls_token=False, extra_tokens=0):
    """
    Generate a 2D sine-cosine positional embedding.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size_h (int): Grid size along the height.
        grid_size_w (int): Grid size along the width.
        cls_token (bool): If True, prepend extra tokens for classification.
        extra_tokens (int): Number of extra tokens to prepend.

    Returns:
        pos_embed (np.ndarray): Positional embedding of shape
                                (grid_size_h*grid_size_w, embed_dim) or
                                (extra_tokens+grid_size_h*grid_size_w, embed_dim) if cls_token is True.
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    # Create a meshgrid (note: grid_w comes first)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sine-cosine positional embedding from a grid.

    Args:
        embed_dim (int): Dimension of the embedding (must be even).
        grid (np.ndarray): Grid tensor of shape (2, 1, grid_size_h, grid_size_w).

    Returns:
        emb (np.ndarray): Positional embedding of shape (grid_size_h*grid_size_w, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."
    # Use half of the dimensions for encoding the height
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sine-cosine positional embedding.

    Args:
        embed_dim (int): Output dimension for each position (must be even).
        pos (np.ndarray): Array of positions to encode of shape (M,).

    Returns:
        emb (np.ndarray): Positional embedding of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # Outer product: (M, D/2)
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, embed_dim)
    return emb
