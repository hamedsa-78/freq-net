import torch


class CharbonnierLoss:
    def __init__(self, epsilon: float = 1e-6):
        self.bc = self.weight_initialization_R10()
        self.epsilon = epsilon

    @staticmethod
    def weight_initialization_R10():
        """
        weight initialization for R = 10 (10 , 10) feature maps
        """
        weights = [1, 1, 1, 1, 5, 10, 10, 5, 1, 1]
        weight_tensors = torch.ones((10, 10), dtype=torch.float32)
        for i in range(10):
            for j in range(10):
                weight_tensors[i, j] = weights[max(i, j)]
        return weight_tensors

    def __call__(self, x, y):
        """
        Computes the Charbonnier loss between two tensors x and y along with the frequency loss.

        Args:
            x (torch.Tensor): The predicted tensor of shape (*, C).
            y (torch.Tensor): The target tensor same shape as x.
            bc (torch.Tensor): The weights for each channel of x of shape (C,).
            epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

        Returns:
            torch.Tensor: The Charbonnier loss along with the frequency loss.
        """
        assert x.shape == y.shape, "x and y tensors must have the same shape"
        assert (
            x.shape[-1] == self.bc.shape[-1]
        ), "x and y tensors must have the same channel dim "

        diff =  torch.sqrt((x - y).pow(2) + self.epsilon**2)

        # ( B , 1 ,100 ,  32 , 32 ) -> (B , 1 , 32 , 32 , 10 , 10)
        diff = diff.movedim(2, 4).reshape((-1, 1, 32, 32, 10, 10))

        charbonnier = diff * self.bc
        freq_loss = torch.sum(charbonnier) / torch.prod(torch.tensor(x.shape))

        return freq_loss
