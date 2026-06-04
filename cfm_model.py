import torch
import torch.nn as nn
from torchcfm.models.unet import UNetModel
from torchcfm.models.unet.unet import timestep_embedding


class CFMUNet(UNetModel):
    def __init__(self, cond_dim: int, cond_hidden_dim:int = 64, **kwargs):
        super().__init__(**kwargs)
        time_embed_dim = kwargs["num_channels"] * 4
        self.label_emb = nn.Sequential(
            nn.Linear(cond_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, time_embed_dim),
        )

    def forward(self, t, x, y=None):
        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y is not None
            emb = emb + self.label_emb(y)  # y: (batch, cond_dim) float

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
