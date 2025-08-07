import pytorch_lightning as pl
import torch.nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# utils
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import src.architectures.PeepholeConvLSTM
import wandb
import os
from omegaconf import OmegaConf
from datetime import datetime

# Custom
from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR_data_loader import ConvLSTMSevirDataModule
from src.utils.Logger import ImagePredictionLogger

OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)

    dm = ConvLSTMSevirDataModule(
        step=cfg.data.step,
        width=cfg.data.width,
        height=cfg.data.height,
        sequence_length=cfg.data.sequence_length,
        files_dir=cfg.data.dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_files_percent=cfg.data.train_files_percent,
        val_files_percent=cfg.data.val_files_percent,
        test_files_percent=cfg.data.test_files_percent
    )

    dm.setup('fit')
    dm.setup('test')
    val_loader = dm.val_dataloader()

    early_stop_callback = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        verbose=cfg.early_stopping.verbose,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        dirpath="checkpoints",
        filename=cfg.checkpoint.filename,
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode
    )

    # CHANGED: instantiate(cfg.model) now returns a valid RainPredictor (LightningModule).
    main_model = instantiate(cfg.model)

    val_samples = next(iter(val_loader))
    trainer = pl.Trainer(
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            ImagePredictionLogger(val_samples, num_samples=cfg.trainer.num_visualised_samples)
        ],
        logger=WandbLogger(
            project=cfg.wandb.project_name,
            job_type='train',
            name=cfg.experiment_id,
            config=OmegaConf.to_container(cfg, resolve=True)
        ),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs
    )

    trainer.fit(main_model, dm)
    trainer.test(model=main_model, datamodule=dm)
    wandb.finish()

if __name__ == "__main__":
    main()
