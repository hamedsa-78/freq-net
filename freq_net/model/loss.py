import torch


def charbonnier_loss(
    x: torch.Tensor, y: torch.Tensor, bc: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Charbonnier loss between two tensors x and y along with the frequency loss.

    Args:
        x (torch.Tensor): The predicted tensor of shape (B, C, H, W).
        y (torch.Tensor): The target tensor of shape (B, C, H, W).
        bc (torch.Tensor): The weights for each channel of x of shape (C,).
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: The Charbonnier loss along with the frequency loss.
    """
    assert x.shape == y.shape, "x and y tensors must have the same shape"
    assert len(x.shape) == 4, "x and y tensors must have shape of (B, C, H, W)"
    B, C, H, W = x.shape

    assert (
        len(bc.shape) == 1 and bc.shape[-1] == C
    ), "Weights tensor must have the shape of (C,)"

    charbonnier = torch.sqrt((x - y).pow(2) + epsilon) * bc.view(1, -1, 1, 1)
    freq_loss = torch.sum(charbonnier) / torch.prod(torch.tensor(x.shape))

    return freq_loss
