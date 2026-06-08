
import math
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional
import functools


class SinusoidalPosEmb(nnx.Module):

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x * 100
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeEmbedding(nnx.Module):

    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(1, dim, rngs=rngs)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = x * jnp.tanh(jnp.where(x < 20, jnp.log1p(jnp.exp(x)), x))  # Mish with numerical stability
        x = self.linear2(x)
        return x


class ContinuousCondEmbedder(nnx.Module):

    def __init__(self, attr_dim: int, hidden_size: int, rngs: nnx.Rngs):
        self.attr_dim = attr_dim
        self.embedding = nnx.Linear(attr_dim, attr_dim * 128, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=128,
            decode=False,
            rngs=rngs
        )
        self.linear = nnx.Linear(128 * attr_dim, hidden_size, rngs=rngs)

    def __call__(self, attr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        emb = self.embedding(attr).reshape((-1, self.attr_dim, 128))
        if mask is not None:
            emb = emb * jnp.expand_dims(mask, -1)
        emb = self.attn(emb)
        emb = emb.reshape(emb.shape[0], -1)
        return self.linear(emb)


def modulate(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)


class DiTBlockJAX(nnx.Module):

    def __init__(self, hidden_size: int, n_heads: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(hidden_size, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=n_heads, in_features=hidden_size, decode=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.mlp_linear1 = nnx.Linear(hidden_size, hidden_size * 4, rngs=rngs)
        self.mlp_linear2 = nnx.Linear(hidden_size * 4, hidden_size, rngs=rngs)
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size * 6, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        adaln_params = self.adaLN_modulation(t)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaln_params, 6, axis=1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x)
        x = x + jnp.expand_dims(gate_msa, 1) * attn_out

        x_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp_linear1(x_mlp)
        mlp_out = nnx.gelu(mlp_out, approximate=True)
        mlp_out = self.mlp_linear2(mlp_out)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_out
        return x


class Finallayer1d(nnx.Module):

    def __init__(self, hidden_size: int, out_dim: int, rngs: nnx.Rngs):
        self.norm_final = nnx.LayerNorm(hidden_size, use_bias=False, use_scale=False, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, out_dim, rngs=rngs)
        self.adaln_linear = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        adaln_out = nnx.silu(t)
        adaln_out = self.adaln_linear(adaln_out)
        shift, scale = jnp.split(adaln_out, 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1d(nnx.Module):

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs, attr_dim: Optional[int] = None, d_model: int = 384,
                 n_heads: int = 6, depth: int = 12):
        self.attr_dim = attr_dim
        self.in_dim, self.d_model, self.n_heads, self.depth = in_dim, d_model, n_heads, depth

        self.x_proj = nnx.Linear(in_dim, d_model, rngs=rngs)
        self.t_emb = TimeEmbedding(d_model, rngs=rngs)
        if attr_dim is not None:
            self.attr_proj = ContinuousCondEmbedder(attr_dim, d_model, rngs=rngs)
        self.pos_emb = SinusoidalPosEmb(d_model)

        self.blocks = [DiTBlockJAX(d_model, n_heads, rngs=rngs) for _ in range(depth)]
        self.final_layer = Finallayer1d(d_model, out_dim, rngs=rngs)

    def initialize_weights(self):
        print("Initializing weights with zero values...")

        for block in self.blocks:
            if hasattr(block, 'adaLN_modulation') and hasattr(block.adaLN_modulation, 'layers'):
                if len(block.adaLN_modulation.layers) > 1:
                    block.adaLN_modulation.layers[1].kernel.value = jnp.zeros_like(
                        block.adaLN_modulation.layers[1].kernel.value
                    )
                    block.adaLN_modulation.layers[1].bias.value = jnp.zeros_like(
                        block.adaLN_modulation.layers[1].bias.value
                    )

        if hasattr(self.final_layer, 'adaln_linear'):
            self.final_layer.adaln_linear.kernel.value = jnp.zeros_like(
                self.final_layer.adaln_linear.kernel.value
            )
            self.final_layer.adaln_linear.bias.value = jnp.zeros_like(
                self.final_layer.adaln_linear.bias.value
            )

        if hasattr(self.final_layer, 'linear'):
            self.final_layer.linear.kernel.value = jnp.zeros_like(
                self.final_layer.linear.kernel.value
            )
            self.final_layer.linear.bias.value = jnp.zeros_like(
                self.final_layer.linear.bias.value
            )

    def __call__(self,
                 x: jnp.ndarray,
                 t: jnp.ndarray,
                 attr: Optional[jnp.ndarray] = None,
                 mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        pos_emb = self.pos_emb(jnp.arange(x.shape[1]))
        x = self.x_proj(x) + jnp.expand_dims(pos_emb, 0)
        t = self.t_emb(t)
        if attr is not None:
            assert self.attr_dim is not None, "Model is not conditional and cannot accept attributes"
            t = t + self.attr_proj(attr, mask)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return jnp.tanh(x)
