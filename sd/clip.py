import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbeddings(nn.Module):
    def __init__(self, n_vocab:int, n_embds:int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embds)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embds))

    def forward(self, tokens):
        # (Batch_Size, seq_len) => (Batch_Size, seq_len, Dim)
        x= self.token_embedding(tokens)

        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embds: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embds)
        self.attention = SelfAttention(n_heads, n_embds)
        self.layernorm_2 = nn.LayerNorm(n_embds)
        self.linear_1 = nn.Linear(n_embds, 4 * n_embds)
        self.linear_2 = nn.Linear(4 * n_embds, n_embds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # (Batch_)Size, seq_len, Dim)
        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        ## feedforward layer

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGeLU (https://arxiv.org/abs/2002.04745) activation function

        x = self.linear_2(x)
        x += residue

        return x



class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbeddings(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layerNorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (Batch_Size, seq_len) => (Batch_Size, seq_len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, seq_len, Dim) => (Batch_Size, seq_len, Dim)
        state = self.layerNorm(state)

        return state
    



