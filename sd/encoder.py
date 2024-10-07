import torch    
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        
        super().__init__(

            # (Batch_size, Channel, Height, Width) => (Batch_size, 128, Height, Width)
            nn.Conv2d(3,128,kernel_size=3, padding=1),

            # (Batch_size, 128, Height, Width) => (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) => (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_size, 128, Height/2, Width/2) => (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size, 256, Height/2, Width/2) => (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height/2, Width/2) => (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 256, Height/4, Width/4) => (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size, 512, Height/4, Width/4) => (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/4, Width/4) => (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            
            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),
            
            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),

            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 512, Height/8, Width/8)
            nn.SiLU(),

            # (Batch_size, 512, Height/8, Width/8) => (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_size, 8, Height/8, Width/8) => (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

        
    def forward(self, x, noise):
        # x: (Batch_size, Channel, Height, Width)
        # noise: (Batch_size, out_channels, Height/8, Width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):

                # (Padding_Left, padding_Right, padding_top, padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)
        
        # Its not just a compressed image, but actally a distribution.
        # paramter of multivariance gaussian distribution (mean, variance)

        # (Batch_size, 8, Height/8, Width/8) => two tensors of shape (Batch_size, 4, Height/8, Width/8
        mean, log_variance = torch.chunk(x,2, dim=1)

        # (Batch_size, 4, Height/8, Width/8) => (Batch_size, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)   

        # (Batch_size, 4, Height/8, Width/8) => (Batch_size, 4, Height/8, Width/8)
        variance = log_variance.exp()

        # (Batch_size, 4, Height/8, Width/8) => (Batch_size, 4, Height/8, Width/8)
        stdev = variance.sqrt()

        # Z= N(0,1) -> N(mean, variance)=X
        # X = mean + stddev * Z
        x = mean + stdev * noise

        # Scale the output of constant
        x *= 0.18215
        return x
        

