from dotenv import load_dotenv

load_dotenv() 

import comet_ml 

import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import HiFiGANTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(model)

    def count_params(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    total, trainable = count_params(model)
    logger.info(f"Total parameters:     {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Non-trainable:        {total - trainable:,}")

    if hasattr(model, 'generator'):
        g_total, g_train = count_params(model.generator)
        logger.info(f"  Generator:          {g_total:,} (trainable: {g_train:,})")

    if hasattr(model, 'discriminator'):
        d_total, d_train = count_params(model.discriminator)
        logger.info(f"  Discriminator:      {d_total:,} (trainable: {d_train:,})")

        loss_function = instantiate(config.loss_function).to(device)
        metrics = instantiate(config.metrics)

    if hasattr(config, 'optimizer_g') and hasattr(config, 'optimizer_d'):
        logger.info("Initializing separate optimizers for generator and discriminator")

        if hasattr(model, 'generator') and hasattr(model, 'discriminator'):
            trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
            trainable_params_d = filter(lambda p: p.requires_grad, model.discriminator.parameters())
        else:
            trainable_params_g = filter(lambda p: p.requires_grad, model.parameters())
            trainable_params_d = []
            logger.warning("Model doesn't have 'generator' and 'discriminator' attributes")

        optimizer_g = instantiate(config.optimizer_g, params=trainable_params_g)
        optimizer_d = instantiate(config.optimizer_d, params=trainable_params_d)

        optimizer = {
            'generator': optimizer_g,
            'discriminator': optimizer_d
        }

        if hasattr(config, 'lr_scheduler_g') and hasattr(config, 'lr_scheduler_d'):
            lr_scheduler_g = instantiate(config.lr_scheduler_g, optimizer=optimizer_g)
            lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)
            lr_scheduler = {
                'generator': lr_scheduler_g,
                'discriminator': lr_scheduler_d
            }
        else:
            lr_scheduler = None
    else:
        logger.info("Initializing single optimizer")
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = instantiate(config.optimizer, params=trainable_params)

        if hasattr(config, 'lr_scheduler'):
            lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
        else:
            lr_scheduler = None

    epoch_len = config.trainer.get("epoch_len")

    trainer = HiFiGANTrainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_d=lr_scheduler_d,
        lr_scheduler_g=lr_scheduler_g,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
