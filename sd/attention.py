import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embeds: int, in_proj_bias: bool = True, out_proj_bias: bool = True):

        super().__init__()

        self.in_proj = nn.Linear(d_embeds, 3*d_embeds, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embeds, d_embeds, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embeds // n_heads

    
    def forward(self, x: torch.Tensor, casual_mask = True):
        
        # x: (Batch_size, Seq_len,  Dim)
        input_shape = x.shape

        batch_size, seq_len, d_embeds = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        # (Batch_size, Seq_len, Dim) => (Batch_size, Seq_len, 3*Dim) => 3 tensors of shape (Batch_size, Seq_len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (Batch_size, Seq_len, Dim) => (Batch_size, Seq_len, H, Dim/H) => (Batch_size, H, Seq_len, Dim/H)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # (Batch_size, H, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # MAsk where upper triangle is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)


        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # (Batch_size, H, Seq_len, Seq_len) @ (Batch_size, H, Seq_len, Dim/H) => (Batch_size, H, Seq_len, Dim/H)
        output = weight @ v

        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (BAtch_size, seq_len, Dim)
        return output





        