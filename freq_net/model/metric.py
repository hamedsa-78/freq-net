import torch


def frm(loss_freq: torch.Tensor) -> torch.Tensor:
    assert loss_freq > 0, "loss_freq could not be zero"
    return 10 * torch.log10(1 / loss_freq)


def psnr(x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    assert x.max() <= max_value
    assert x.min() >= 0
    assert y.max() <= max_value
    assert y.min() >= 0
    assert x.shape == y.shape, "tensors must have the same shape"

    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return 100
    psnr_value = 10 * torch.log10(max_value**2 / mse)
    return psnr_value
