## Trying to build yet another PyTorch implementation of Stable Diffusion.

- VAE AutoEncoder Added 
- Clip Text Encoder Added
- UNET (WIP)
- Scheduler (TO be started)


### Diffusion Model
Diffusion Models are probabilistic models designed to learn a data distribution p(x) by gradually denoising a normally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length T. [High-Resolution Image Synthesis with Latent Diffusion Models-2022](https://arxiv.org/pdf/2112.10752)

The process consisits of 2 steps:

    -- a fixed forward process q, that gradually adds Gaussian noise to an image until we end up with pure noise.
    -- a learned reverse diffusion process pθ, where θ is a parameter of neural network, updated by gradient descent.
     

    ![Diffusion process](https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion_figure.png)










