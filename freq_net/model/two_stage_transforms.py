import torch
import numpy as np
from torch_dct import dct_2d, idct_2d


# (batch_size = 64 , channel = 1 , block = 32 , block = 32 , heigth = 256 , width = 256)


class TwoStageDCT:
    def __init__(self, block_size=32, R=10):
        self.block_size = block_size
        self.R = R

    def dct(self, input: torch.Tensor) -> torch.Tensor:
        # (B , 512 , 512 ) -> (B, block_number = 16 , block_number = 16 , block_size = 32 , block_size = 32)
        batch_size, height, width = input.shape

        # Reshape the images to separate blocks of size block_size * block_size
        blocks = input.unfold(1, self.block_size, self.block_size).unfold(
            2, self.block_size, self.block_size
        )

        blocks = [
            blocks[b, i, j, :, :]
            for b in range(batch_size)
            for i in range(blocks.shape[1])
            for j in range(blocks.shape[2])
        ]

        blocks = torch.stack([dct_2d(block, "ortho") for block in blocks])

        blocks = blocks.reshape(
            batch_size,
            height // self.block_size,
            width // self.block_size,
            self.block_size,
            self.block_size,
        )

        return blocks  # (B , 16, 16 , 32 , 32 )

    def idct(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        # (B , 1 , block = 16, block = 16 , block_size = 32, block_size = 32)
        batch_size, block, _, block_size, _ = dct_coeffs.shape

        blocks = [
            dct_coeffs[b, i, j, :, :]
            for b in range(batch_size)
            for i in range(block)
            for j in range(block)
        ]

        blocks = torch.stack([idct_2d(block, "ortho") for block in blocks])

        blocks = blocks.reshape(batch_size, block, block, block_size, block_size)

        blocks.transpose(2, 3)

        images = blocks.reshape(batch_size, block * block_size, block * block_size)

        return images  # (B , 1 , 512 , 512) original image blocks

    def two_stage_dct_in(self, dct_coeffs) -> torch.Tensor:
        def channel_norm(
            feature_maps: torch.Tensor,
        ) -> torch.Tensor:
            epsilon = 1e-8

            mean = feature_maps.mean(dim=[-2, -1], keepdim=True)
            std = feature_maps.std(dim=[-2, -1], keepdim=True)

            if torch.any(std == 0):
                std = std + epsilon

            normalized_feature_maps = (feature_maps - mean) / std
            return normalized_feature_maps

        feature_maps = dct_coeffs[..., : self.R, : self.R]  # (B  , 16 , 16 , 10 , 10)

        normalized_feature_maps = channel_norm(feature_maps)
        return feature_maps, normalized_feature_maps

    def two_stage_idct_out(
        self,
        lr_image: torch.Tensor,
        dct_low_resolution: torch.Tensor,
        feature_maps: torch.Tensor,
        normalized_feature_maps: torch.Tensor,
    ):
        def channel_denormalization(
            feature_maps: torch.Tensor, normalized_feature_maps: torch.Tensor
        ):
            """
            Reverse the channel-wise normalization performed by channel_norm()
            """
            mean = feature_maps.mean(dim=[-2, -1], keepdim=True)
            std = feature_maps.std(dim=[-2, -1], keepdim=True)

            # Perform channel-wise denormalization
            denormalized_feature_maps = (normalized_feature_maps * std) + mean
            return denormalized_feature_maps

        # First, calculate the Discrete Cosine Transform (DCT) of the 'dct_low_resolution_image'.
        # We will need this DCT coefficients that are not in the top-right 'R * R' (10 * 10) regions of the image.
        # name it 'dct_low_resolution'.
        # Next, merge the 'denormalized_feature_map' with 'dct_low_resolution'.
        # Finally, perform the Inverse Discrete Cosine Transform (IDCT).

        # (B  , 16 , 16 , 10 , 10)
        denormalized_feature_map = channel_denormalization(
            feature_maps, normalized_feature_maps
        )

        # (B  , 16 , 16 , 32 , 32)
        # dct_low_resolution = self.dct(self.input)

        dct_coeffs = dct_low_resolution.clone()

        dct_coeffs[..., : self.R, : self.R] = denormalized_feature_map

        high_resolution_image = self.idct(dct_coeffs)  # (B , 512 , 512)

        three_channel_hr_image = lr_image.clone()  # (B , 3 , 512 , 512)
        three_channel_hr_image[:, 0, ...] = high_resolution_image
        return three_channel_hr_image  # (B, 3 , 512 , 512)
