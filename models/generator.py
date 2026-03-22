import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler


class StyleTextGenerator(nn.Module):
    """
    Diffusion generator conditioned on style and text embeddings.
    Args:
        style_dim  (int): style embedding size — 512
        text_dim   (int): text embedding size  — 256
        fusion_dim (int): shared fusion dimension
    """
    def __init__(self, style_dim=512, text_dim=256, fusion_dim=256):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.style_proj = nn.Linear(style_dim, fusion_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        self.unet = UNet2DConditionModel(
            sample_size=(128, 512),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=fusion_dim,
            norm_num_groups=32,
        )

    def fuse(self, style_emb, text_emb):
        """
        F_fused = alpha*style_proj + beta*text_emb  (alpha+beta=1 via softmax)
        Returns: (B, 1, fusion_dim)
        """
        weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        style_proj = self.style_proj(style_emb)
        fused = weights[0] * style_proj + weights[1] * text_emb
        return fused.unsqueeze(1)

    def forward(self, noisy_image, timestep, style_emb, text_emb):
        """
        Args:
            noisy_image (B, 1, 128, 512)
            timestep    (B,)
            style_emb   (B, 512)
            text_emb    (B, 256)
        Returns:
            predicted noise (B, 1, 128, 512)
        """
        conditioning = self.fuse(style_emb, text_emb)
        return self.unet(noisy_image, timestep, conditioning).sample


if __name__ == "__main__":
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")
    generator = StyleTextGenerator().to(device)
    B = 2
    noisy_image = torch.randn(B, 1, 128, 512).to(device)
    timestep    = torch.randint(0, 1000, (B,)).to(device)
    style_emb   = torch.randn(B, 512).to(device)
    text_emb    = torch.randn(B, 256).to(device)
    output = generator(noisy_image, timestep, style_emb, text_emb)
    print(f"Input:  {noisy_image.shape}")
    print(f"Output: {output.shape}")
    print(f"alpha={torch.sigmoid(generator.alpha).item():.3f}  "
          f"beta={torch.sigmoid(generator.beta).item():.3f}")