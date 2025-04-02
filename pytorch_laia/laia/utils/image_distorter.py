import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Optional
import math

class ImageDistorter(nn.Module):
    def __init__(
        self,
        max_rotation: float = 5.0,  # degrees
        max_scale: float = 0.2,
        max_shear: float = 0.2,
        max_translation: float = 0.1,
        random_elastic: bool = True,
        elastic_sigma: float = 6.0,
        elastic_alpha: float = 40.0,
    ):
        super().__init__()
        self.max_rotation = max_rotation
        self.max_scale = max_scale
        self.max_shear = max_shear
        self.max_translation = max_translation
        self.random_elastic = random_elastic
        self.elastic_sigma = elastic_sigma
        self.elastic_alpha = elastic_alpha

    def get_random_affine_params(
        self, batch_size: int, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random rotation
        angle = torch.empty(batch_size).uniform_(-self.max_rotation, self.max_rotation)
        
        # Random scale
        scale = torch.empty(batch_size).uniform_(
            1 - self.max_scale, 1 + self.max_scale
        )
        
        # Random shear
        shear = torch.empty(batch_size, 2).uniform_(-self.max_shear, self.max_shear)
        
        # Random translation
        translate = torch.empty(batch_size, 2).uniform_(
            -self.max_translation, self.max_translation
        )
        translate *= torch.tensor([width, height])
        
        return angle, scale, shear, translate

    def elastic_deformation_grid(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if not self.random_elastic:
            return None

        # Create random displacement fields
        dx = torch.randn(batch_size, height, width, device=device) * self.elastic_alpha
        dy = torch.randn(batch_size, height, width, device=device) * self.elastic_alpha

        # Apply Gaussian filter
        dx = F.gaussian_blur(dx.unsqueeze(1), kernel_size=3, sigma=self.elastic_sigma).squeeze(1)
        dy = F.gaussian_blur(dy.unsqueeze(1), kernel_size=3, sigma=self.elastic_sigma).squeeze(1)

        # Create sampling grid
        x = torch.arange(width, device=device).view(1, 1, -1).repeat(batch_size, height, 1)
        y = torch.arange(height, device=device).view(1, -1, 1).repeat(batch_size, 1, width)

        x = x + dx
        y = y + dy

        # Normalize coordinates to [-1, 1]
        x = 2 * x / (width - 1) - 1
        y = 2 * y / (height - 1) - 1

        return torch.stack([x, y], dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        device = x.device

        # Get affine parameters
        angle, scale, shear, translate = self.get_random_affine_params(batch_size, height, width)
        
        # Apply affine transformation
        x = F.affine(
            x,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=F.InterpolationMode.BILINEAR
        )

        # Apply elastic deformation if enabled
        if self.random_elastic:
            grid = self.elastic_deformation_grid(batch_size, height, width, device)
            x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')

        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ImageDistorter")
        parser.add_argument("--max_rotation", type=float, default=5.0)
        parser.add_argument("--max_scale", type=float, default=0.2)
        parser.add_argument("--max_shear", type=float, default=0.2)
        parser.add_argument("--max_translation", type=float, default=0.1)
        parser.add_argument("--random_elastic", type=bool, default=True)
        parser.add_argument("--elastic_sigma", type=float, default=6.0)
        parser.add_argument("--elastic_alpha", type=float, default=40.0)
        return parent_parser 