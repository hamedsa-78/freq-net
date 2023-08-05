import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def frm(loss_freq: torch.Tensor) -> torch.Tensor:
    assert loss_freq > 0, "loss_freq could not be zero"
    return 10 * torch.log10(1 / loss_freq)


def psnr(x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    assert x.shape == y.shape, "tensors must have the same shape"
    mse = torch.mean((x - y) ** 2)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr
