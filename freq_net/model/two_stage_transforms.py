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
        self.input = input  # upscaled input (b , 1 , 512 , 512)
        self.block_size = block_size
        self.R = R
        self.batch = batch
        self.channel = channel

    def dct(self, input: torch.Tensor) -> torch.Tensor:
        #
        # (Batch , 1 just Y channel , 256 , 256 ) ->
        # (Batch , 1 , block_number = 16 , block_number = 16 , block_size = 32 , block_size =32)
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
        return dct_blocks  # (B , 1 , 16 , 16 , 32 , 32 )

        # Reshape the blocks back to the original image shape
        # dct_images = dct_blocks.reshape(
        #     batch_size,
        #     channels,
        #     height // self.block_size,
        #     width // self.block_size,
        #     self.block_size,
        #     self.block_size,
        # )
        # dct_images = dct_images.transpose(4, 3).reshape(
        #     batch_size, channels, height, width
        # )
        # return dct_images

    def idct(self, dct_coeffs: torch.Tensor) -> torch.Tensor:
        # (B , 1 , block, block , 10, 10)
        batch_size, channels, block, _, block_size, _ = dct_coeffs.shape

        # blocks = dct_coeffs.reshape(
        #     batch_size,
        #     channels,
        #     height // self.block_size,
        #     self.block_size,
        #     width // self.block_size,
        #     self.block_size,
        # )
        # blocks = (
        #     blocks.transpose(4, 3)
        #     .reshape(-1, channels, self.block_size, self.block_size)
        #     .numpy()
        # )
        # Perform DCT on each block
        idct_blocks = torch.from_numpy(
            fftpack.idct(
                fftpack.idct(dct_coeffs, axis=-2, norm="ortho"), norm="ortho", axis=-1
            )
        )
        # Reshape the blocks back to the original image shape
        dct_images = idct_blocks.reshape(
            batch_size,
            channels,
            block,
            block,
            self.block_size,
            self.block_size,
        )
        dct_images = dct_images.transpose(4, 3).reshape(
            batch_size, channels, block * self.block_size, block * self.block_size
        )
        return idct_blocks  # (B , 1 , 512 , 512) original image blocks

    def two_stage_dct_in(self) -> torch.Tensor:
        def channel_norm(
            feature_maps: torch.Tensor,
        ) -> torch.Tensor:  # in : (B , 1 , 16 , 16 , 32 , 32)
            epsilon = 1e-8

            mean = feature_maps.mean(dim=[-2, -1], keepdim=True)
            std = feature_maps.std(dim=[-2, -1], keepdim=True)

            if torch.any(std == 0):
                std = std + epsilon

            normalized_feature_maps = (feature_maps - mean) / std
            return normalized_feature_maps

        dct_coeffs = self.dct(self.input)  # (B , 1 , 16, 16 , 32 , 32 )
        feature_maps = dct_coeffs[..., self.R, : self.R]  # (B , 1 , 16 , 16 , 10 , 10)

        normalized_feature_maps = channel_norm(feature_maps)
        return feature_maps, normalized_feature_maps

    def two_stage_idct_out(
        self,
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

        # (B , 1 , 16 , 16 , 10 , 10)
        denormalized_feature_map = channel_denormalization(
            feature_maps, normalized_feature_maps
        )

        # (B , 1 , 16 , 16 , 32 , 32)
        # dct_low_resolution = self.dct(self.input)

        dct_coeffs = dct_low_resolution.clone()

        dct_coeffs[..., self.R, : self.R] = denormalized_feature_map

        high_resolution_image = self.idct(dct_coeffs)
        return high_resolution_image  # (B, 1 , 512 , 512)


def channel_norm(
    feature_maps: torch.Tensor,
) -> torch.Tensor:  # in : (B , 1 , 16 , 16 , 32 , 32)
    epsilon = 1e-8

    mean = feature_maps.mean(dim=[-2, -1], keepdim=True)
    std = feature_maps.std(dim=[-2, -1], keepdim=True)

    if torch.any(std == 0):
        std = std + epsilon

    normalized_feature_maps = (feature_maps - mean) / std
    return normalized_feature_maps


def two_stage_idct_out(
    up_scaled_lr_image,  # (B , 3 , 512 , 512)
    dct_low_resolution: torch.Tensor,  # (B , 1 , 16 , 16 , 10 , 10 )
    feature_maps: torch.Tensor,  # (B , 1 , 16 , 16 , 10 , 10 ) before normalization
    normalized_feature_maps: torch.Tensor,  # (B , 1 , 16 , 16 , 10 , 10 )
):
    def idct(dct_coeffs: torch.Tensor) -> torch.Tensor:
        # (B , 1 , block, block , 10, 10)
        batch_size, channels, block, _, block_size, _ = dct_coeffs.shape

        idct_blocks = torch.from_numpy(
            fftpack.idct(
                fftpack.idct(dct_coeffs, axis=-2, norm="ortho"), norm="ortho", axis=-1
            )
        )
        # Reshape the blocks back to the original image shape
        dct_images = idct_blocks.reshape(
            batch_size,
            channels,
            block,
            block,
            block_size,
            block_size,
        )
        dct_images = dct_images.transpose(4, 3).reshape(
            batch_size, channels, block * block_size, block * block_size
        )
        return idct_blocks  # (B , 1 , 512 , 512) original image blocks

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

    # (B , 1 , 16 , 16 , 10 , 10)
    denormalized_feature_map = channel_denormalization(
        feature_maps, normalized_feature_maps
    )

    region_size = denormalized_feature_map.shape[-1]

    # (B , 1 , 16 , 16 , 32 , 32)
    # dct_low_resolution = self.dct(self.input)

    dct_coeffs = dct_low_resolution.clone()

    dct_coeffs[..., region_size, :region_size] = denormalized_feature_map

    high_resolution_image = idct(dct_coeffs)

    three_channel_high_resolution = up_scaled_lr_image.clone()
    three_channel_high_resolution[:, 0, ...] = high_resolution_image

    return three_channel_high_resolution  # (B, 1 , 512 , 512)
