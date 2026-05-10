import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        embed_dim: model dimension (must be divisible by num_heads)
        num_heads: number of attention heads
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = (embed_dim) // num_heads

        # linear projections for q, k, v and output
        self.q_proj = nn.Linear(embed_dim // 2, embed_dim // 2)
        self.k_proj = nn.Linear(embed_dim // 2, embed_dim // 2)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)


    def _split_heads_1(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k // 2)
        return x.transpose(1, 2)
    
    def _split_heads_2(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, heads, seq_len, d_k) -> (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, seq_len, heads, d_k)
        b, seq, _, _ = x.size()
        return x.contiguous().view(b, seq, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query/key/value: (batch, seq_len, embed_dim)
        mask: optional mask with shape broadcastable to (batch, num_heads, seq_q, seq_k)
              You can provide:
              - None
              - mask of shape (batch, seq_q, seq_k) (bool or float additive)
              - mask of shape (seq_q, seq_k)
        """
        # linear projections
        # print('qkv', query.shape, key.shape, value.shape)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # split into heads
        q = self._split_heads_1(q)  # (batch, heads, seq_q, d_k)
        k = self._split_heads_1(k)  # (batch, heads, seq_k, d_k)
        v = self._split_heads_2(v)  # (batch, heads, seq_v, d_k)

        d_k = q.size(-1)
        # (..., seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / int(math.sqrt(d_k)) # b, h, seq_q, seq_k

        b, h, seq_q, seq_k = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
        hw_q, hw_k = int(math.sqrt(seq_q)), int(math.sqrt(seq_k))

        scores = scores.contiguous().view(b, h, hw_q, hw_q, hw_k, hw_k)
        scores = scores.contiguous().view(b, h, hw_q // 2, 2, hw_q // 2, 2, hw_k // 2, 2, hw_k // 2, 2)
        try:
            scores = scores.mean(dim=(3,5,7,9))   
        except TypeError:
            for d in sorted((3,5,7,9), reverse=True):
                scores = scores.mean(dim=d)
        scores = scores.contiguous().view(b, h, seq_q // 4, seq_k // 4)

        attn = F.softmax(scores, dim=-1)
        
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v)

        # combine heads
        attn_output = self._combine_heads(output)  # (batch, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output
    
class MultiHeadAttention_dino(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        embed_dim: model dimension (must be divisible by num_heads)
        num_heads: number of attention heads
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = (embed_dim) // num_heads

        # linear projections for q, k, v and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)


    def _split_heads_1(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def _split_heads_2(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim) -> (batch, heads, seq_len, d_k)
        b, seq, _ = x.size()
        x = x.view(b, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, heads, seq_len, d_k) -> (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, seq_len, heads, d_k)
        b, seq, _, _ = x.size()
        return x.contiguous().view(b, seq, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query/key/value: (batch, seq_len, embed_dim)
        mask: optional mask with shape broadcastable to (batch, num_heads, seq_q, seq_k)
              You can provide:
              - None
              - mask of shape (batch, seq_q, seq_k) (bool or float additive)
              - mask of shape (seq_q, seq_k)
        """
        # linear projections
        # print('qkv', query.shape, key.shape, value.shape)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # split into heads
        q = self._split_heads_1(q)  # (batch, heads, seq_q, d_k)
        k = self._split_heads_1(k)  # (batch, heads, seq_k, d_k)
        v = self._split_heads_2(v)  # (batch, heads, seq_v, d_k)

        d_k = q.size(-1)
        # (..., seq_q, seq_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / int(math.sqrt(d_k)) # b, h, seq_q, seq_k

        # b, h, seq_q, seq_k = scores.shape[0], scores.shape[1], scores.shape[2], scores.shape[3]
        # hw_q, hw_k = int(math.sqrt(seq_q)), int(math.sqrt(seq_k))

        # scores = scores.contiguous().view(b, h, hw_q, hw_q, hw_k, hw_k)
        # scores = scores.contiguous().view(b, h, hw_q // 2, 2, hw_q // 2, 2, hw_k // 2, 2, hw_k // 2, 2)
        # try:
        #     scores = scores.mean(dim=(3,5,7,9))   
        # except TypeError:
        #     for d in sorted((3,5,7,9), reverse=True):
        #         scores = scores.mean(dim=d)
        # scores = scores.contiguous().view(b, h, seq_q // 4, seq_k // 4)

        attn = F.softmax(scores, dim=-1)
        
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v)

        # combine heads
        attn_output = self._combine_heads(output)  # (batch, seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.proj_dropout(attn_output)

        return attn_output
    
class PositionwiseFFN(nn.Module):
    def __init__(self, embed_dim: int, ffn_factor: int, dropout: float = 0.0, activation=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_factor * embed_dim)
        self.fc2 = nn.Linear(ffn_factor * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class KGFPN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_factor: int = 2,
        dropout: float = 0.1,
        activation=nn.GELU,
    ):
        """
        embed_dim: model dimension
        num_heads: attention heads
        ffn_dim: hidden dim in FFN
        dropout: dropout for attention & FFN
        norm_first: if True use Pre-LN (LayerNorm before sublayer), else Post-LN
        """
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(embed_dim, ffn_factor, dropout=dropout, activation=activation)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        
        x = self.mha(q, k, v)
        x = self.norm(x + v)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm1(x)

        return x
    

class KGFPN_dino(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_factor: int = 2,
        dropout: float = 0.1,
        activation=nn.GELU,
    ):
        """
        embed_dim: model dimension
        num_heads: attention heads
        ffn_dim: hidden dim in FFN
        dropout: dropout for attention & FFN
        norm_first: if True use Pre-LN (LayerNorm before sublayer), else Post-LN
        """
        super().__init__()
        self.mha = MultiHeadAttention_dino(embed_dim, num_heads, dropout=dropout)
        self.ffn = PositionwiseFFN(embed_dim, ffn_factor, dropout=dropout, activation=activation)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        
        x = self.mha(q, k, v)
        x = self.norm(x + v)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm1(x)

        return x