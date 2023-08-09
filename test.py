import argparse
import torch
from tqdm import tqdm

import freq_net.data_loader.data_loaders as module_data
import freq_net.model.loss as module_loss
import freq_net.model.metric as module_metric
import freq_net.model.model as module_arch

from parse_config import ConfigParser


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        train=False,
        num_workers=0,
    )

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # config.resume : *.pth
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])(device=device)
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            lr_img, lr_dct = data
            hr_img, hr_dct = target
            lr_img, lr_dct, hr_img, hr_dct = (
                lr_img.to(device),
                lr_dct.to(device),
                hr_img.to(device),
                hr_dct.to(device),
            )

            output, hr_predicted = model(lr_img, lr_dct)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, hr_dct)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(hr_predicted, hr_img) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
