import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embds: int):
        super().__init__()
        
        self.linear_1 = nn.Linear(n_embds, n_embds)
        self.linear_2 = nn.Linear(4 * n_embds, 4 * n_embds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (1, 320)
        x = self.linear_1(x) 

        x = F.silu(x)

        x = self.linear_2(x)

        # x: (1, 1280)
        return x
    

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

         

class Diffusion(nn.Module):
    def __init__(self):
        
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):

        # latent: (Batch_size, 4 , Height/8, width/8)
        # Context: (Batch_size, seq_len, dim)
        # time: (1,320)
        
        time = self.time_embedding(time)

        output = self.unet(latent, context, time)

        output = self.final(output)

        return output