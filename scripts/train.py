#!/usr/bin/env python3

import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import logging
import coloredlogs
from image_wizard.models.lightning.classification_module import ClassificationModule
from image_wizard.utils.cifar_datamodule import CIFAR10DataModule


logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function using PyTorch Lightning."""

    logger.info("CIFAR-10 Training with Hydra + Lightning")

    # Set random seeds for reproducibility
    L.seed_everything(cfg.get("seed", 42))

    # 1. CREATE DATA MODULE
    logger.debug("Setting up data...")
    datamodule = hydra.utils.instantiate(cfg.data)
    logger.debug(f"DataModule: {datamodule.__class__.__name__}")

    # 2. CREATE MODEL
    logger.debug("Creating model...")
    base_model = hydra.utils.instantiate(cfg.model)

    # Wrap in Lightning module
    lightning_module = ClassificationModule(model=base_model, **cfg.training)
    logger.info(f"Model: {base_model.__class__.__name__}")

    # 3. SETUP CALLBACKS
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.training.callbacks]

    # 4. SETUP LOGGER
    logger = CSVLogger("logs", name="cifar_experiment")

    # 5. CREATE TRAINER
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",  # Automatically use GPU if available
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        **cfg.get("trainer", {}),
    )

    # 6. TRAIN
    logger.info("Starting training...")
    trainer.fit(lightning_module, datamodule)

    # 7. TEST
    logger.info("Running final test...")
    trainer.test(lightning_module, datamodule, ckpt_path="best")

    logger.info(f"\nTraining completed!")
    logger.info(f"Best model saved to: {checkpoint_callback.best_model_path}")
    logger.info(f"Logs saved to: logs/cifar_experiment")


if __name__ == "__main__":
    train()
