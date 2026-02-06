import torch.nn as nn

from app.config import device
from app.residual_block.resnet import ResidualBlock


class UpscaleModel(nn.Module):
    def __init__(self, num_blocks: int=32, scale_factor: int=4):
        super(UpscaleModel, self).__init__()
        self.scale_factor: int = scale_factor
        self.num_blocks: int = num_blocks

        # Initial convolutional layer to extract features
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )

        # Stack of residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlock().to(device))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Convolutional layers after the residual blocks
        self.after_res_blocks = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, device=device),
            nn.BatchNorm2d(64)
        )

        # Upsampling using sub-pixel convolutions
        upconv_layers = []
        current_scale = 1
        while current_scale < scale_factor:
            upconv_layers.extend([
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ])
            current_scale *= 2
        self.upconv_layers = nn.Sequential(*upconv_layers)

        # Final convolution to output RGB image
        self.final_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        initial_features = self.first_conv(x)
        res_out = self.res_blocks(initial_features)
        post_res_out = self.after_res_blocks(res_out)
        final_features = initial_features + post_res_out
        upsampled = self.upconv_layers(final_features)
        output = self.final_conv(upsampled)
        return output
