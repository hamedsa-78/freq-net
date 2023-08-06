import torch


class TwoStageDCT:
    def __init__(self, input, block_size=32, R=10):
        self.input = input  # upscaled input
        self.block_size = block_size
        self.R = R

    def two_stage_dct_in(self) -> torch.Tensor:
        def dct(input: torch.Tensor) -> torch.Tensor:
            # (Batch , 1 just Y channel , 256 , 256 ) -> (Batch , block_size = 64 , flattened feature maps = 100)
            # Implement DCT calculation here
            pass  # TODO

        def channel_norm(feature_maps: torch.Tensor) -> torch.Tensor:
            epsilon = 1e-8

            mean = feature_maps.mean(dim=-1, keepdim=True)
            std = feature_maps.std(dim=-1, keepdim=True)

            if torch.any(std == 0):
                std = std + epsilon

            normalized_feature_maps = (feature_maps - mean) / std
            return normalized_feature_maps

        feature_maps = dct(self.input)
        normalized_feature_maps = channel_norm(feature_maps)
        return feature_maps, normalized_feature_maps

    def two_stage_dct_out(
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

        def idct(
            denormalized_feature_map: torch.Tensor,
            dct_low_resolution: torch.Tensor,
        ):
            # First, calculate the Discrete Cosine Transform (DCT) of the 'dct_low_resolution_image'.
            # We will need this DCT coefficients that are not in the top-right 'R * R' (10 * 10) regions of the image.
            # name it 'dct_low_resolution'.
            # Next, merge the 'denormalized_feature_map' with 'dct_low_resolution'.
            # Finally, perform the Inverse Discrete Cosine Transform (IDCT).
            pass  # TODO:

        denormalized_feature_map = channel_denormalization(
            feature_maps, normalized_feature_maps
        )
        high_resolution_image = idct(denormalized_feature_map, self.input)
        return high_resolution_image
