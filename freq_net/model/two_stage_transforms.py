import torch
import numpy as np
from scipy import fftpack

# (batch_size = 64 , channel = 1 , block = 32 , block = 32 , heigth = 256 , width = 256)


class TwoStageDCT:
    def __init__(self, input, block_size=32, R=10):
        batch, channel, height, width = input.shape
        assert R <= min(
            height, width
        ), "R region must smaller than image height and width"
        self.input = input  # upscaled input
        self.block_size = block_size
        self.R = R
        self.batch = batch
        self.channel = channel

    def dct(self, input: torch.Tensor) -> torch.Tensor:
        #
        # (Batch , 1 just Y channel , 256 , 256 ) -> (Batch , block_size = 64 , flattened feature maps = 100)
        batch_size, channels, height, width = input.shape
        # Reshape the images to separate blocks of size block_size * block_size
        blocks = input.reshape(
            batch_size,
            channels,
            height // self.block_size,
            self.block_size,
            width // self.block_size,
            self.block_size,
        )
        blocks = (
            blocks.transpose(4, 3)
            .reshape(-1, channels, self.block_size, self.block_size)
            .numpy()
        )
        # Perform DCT on each block
        dct_blocks = torch.from_numpy(
            fftpack.dct(
                fftpack.dct(blocks, norm="ortho", axis=-2), norm="ortho", axis=-1
            )
        )
        # Reshape the blocks back to the original image shape
        dct_images = dct_blocks.reshape(
            batch_size,
            channels,
            height // self.block_size,
            width // self.block_size,
            self.block_size,
            self.block_size,
        )
        dct_images = dct_images.transpose(4, 3).reshape(
            batch_size, channels, height, width
        )
        return dct_images

    def idct(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        # (Batch , 1 just Y channel , 256 , 256 ) -> (Batch , block_size = 64 , flattened feature maps = 100)
        batch_size, channels, height, width = dct_coeffs.shape
        # Reshape the images to separate blocks of size block_size * block_size
        blocks = dct_coeffs.reshape(
            batch_size,
            channels,
            height // self.block_size,
            self.block_size,
            width // self.block_size,
            self.block_size,
        )
        blocks = (
            blocks.transpose(4, 3)
            .reshape(-1, channels, self.block_size, self.block_size)
            .numpy()
        )
        # Perform DCT on each block
        dct_blocks = torch.from_numpy(
            fftpack.idct(
                fftpack.idct(blocks, axis=-2, norm="ortho"), norm="ortho", axis=-1
            )
        )
        # Reshape the blocks back to the original image shape
        dct_images = dct_blocks.reshape(
            batch_size,
            channels,
            height // self.block_size,
            width // self.block_size,
            self.block_size,
            self.block_size,
        )
        dct_images = dct_images.transpose(4, 3).reshape(
            batch_size, channels, height, width
        )
        return dct_images

    def two_stage_dct_in(self) -> torch.Tensor:
        def channel_norm(feature_maps: torch.Tensor) -> torch.Tensor:
            epsilon = 1e-8

            mean = feature_maps.mean(dim=-1, keepdim=True)
            std = feature_maps.std(dim=-1, keepdim=True)

            if torch.any(std == 0):
                std = std + epsilon

            normalized_feature_maps = (feature_maps - mean) / std
            return normalized_feature_maps

        dct_coeffs = self.dct(self.input)  # (B , 1 , height, width )
        feature_maps = dct_coeffs[:, :, : self.R, : self.R].reshape(
            self.batch, self.channel, -1
        )
        normalized_feature_maps = channel_norm(feature_maps)
        return feature_maps, normalized_feature_maps

    def two_stage_idct_out(
        self, feature_maps: torch.Tensor, normalized_feature_maps: torch.Tensor
    ) -> torch.Tensor:
        def channel_denormalization(
            feature_maps: torch.Tensor, normalized_feature_maps: torch.Tensor
        ) -> torch.Tensor:
            """
            Reverse the channel-wise normalization performed by channel_norm()
            """
            mean = feature_maps.mean(dim=-1, keepdim=True)
            std = feature_maps.std(dim=-1, keepdim=True)

            # Perform channel-wise denormalization
            denormalized_feature_maps = (normalized_feature_maps * std) + mean
            return denormalized_feature_maps

        # First, calculate the Discrete Cosine Transform (DCT) of the 'dct_low_resolution_image'.
        # We will need this DCT coefficients that are not in the top-right 'R * R' (10 * 10) regions of the image.
        # name it 'dct_low_resolution'.
        # Next, merge the 'denormalized_feature_map' with 'dct_low_resolution'.
        # Finally, perform the Inverse Discrete Cosine Transform (IDCT).

        # (B , 1 , 100)
        denormalized_feature_map = channel_denormalization(
            feature_maps, normalized_feature_maps
        )

        # (B , 1 , 10 , 10)
        denormalized_feature_map = denormalized_feature_map.reshape(
            self.batch, self.channel, self.R, self.R
        )

        dct_low_resolution = self.dct(self.input)  # (B , 1 , h = 256 , w = 256)

        dct_coeffs = dct_low_resolution

        dct_coeffs[:, :, : self.R, : self.R] = denormalized_feature_map

        high_resolution_image = self.idct(dct_coeffs)
        return high_resolution_image  # (B, 1 , 256 , 256)
