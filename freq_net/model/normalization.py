import torch


def channel_norm(feature_maps: torch.Tensor) -> torch.Tensor:
    """
    channel_wise normalization of feature_maps on the last channel
    """
    epsilon = 1e-8

    mean = feature_maps.mean(dim=-1, keepdim=True)
    std = feature_maps.std(dim=-1, keepdim=True)

    if torch.any(std == 0):
        std = std + epsilon

    # Perform channel-wise normalization
    normalized_feature_maps = (feature_maps - mean) / std
    return normalized_feature_maps
