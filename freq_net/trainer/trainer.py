import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional

from pathlib import Path
import sys

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(f"{str(root)}/../")

from freq_net.base import BaseTrainer
from freq_net.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns]
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            _, lr_img, lr_dct = data
            hr_rgb, hr_img, hr_dct = target

            lr_img, lr_dct, hr_img, hr_dct = (
                lr_img.to(self.device),
                lr_dct.to(self.device),
                hr_img.to(self.device),
                hr_dct.to(self.device),
            )

            self.optimizer.zero_grad()
            output, hr_predicted_img = self.model(lr_img, lr_dct)
            loss = self.criterion(output, hr_dct)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update("loss", loss.item())

            if hr_predicted_img is not None:
                with torch.no_grad():
                    for index, img_ycrcb in enumerate(hr_predicted_img):
                        for met in self.metric_ftns:
                            if met.__name__ == "psnr":
                                self.train_metrics.update(
                                    met.__name__,
                                    met(img_ycrcb[:, 0, ...], hr_img[index][:, 0, ...]),
                                )
                            elif met.__name__ == "bicubic_psnr":
                                self.train_metrics.update(
                                    met.__name__,
                                    met(
                                        lr_img[index][:, 0, ...],
                                        hr_img[index][:, 0, ...],
                                    ),
                                )

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()  # {loss : 0.1 , psnr  : 0.5 , frm : 0.4}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                _, lr_img, lr_dct = data
                hr_rgb, hr_img, hr_dct = target

                lr_img, lr_dct, hr_rgb, hr_img, hr_dct = (
                    lr_img.to(self.device),
                    lr_dct.to(self.device),
                    hr_rgb.to(self.device),
                    hr_img.to(self.device),
                    hr_dct.to(self.device),
                )

                output, hr_predicted_img = self.model(lr_img, lr_dct)

                loss = self.criterion(output, hr_dct)

                self.valid_metrics.update("loss", loss.item())

                if hr_predicted_img is not None:
                    for index, img_ycrcb in enumerate(hr_predicted_img):
                        for met in self.metric_ftns:
                            if met.__name__ == "psnr":
                                self.valid_metrics.update(
                                    met.__name__,
                                    met(
                                        img_ycrcb[:, 0, ...],
                                        hr_img[index][:, 0, ...],
                                    ),
                                )
                            elif met.__name__ == "bicubic_psnr":
                                self.valid_metrics.update(
                                    met.__name__,
                                    met(
                                        lr_img[index][:, 0, ...],
                                        hr_img[index][:, 0, ...],
                                    ),
                                )

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
