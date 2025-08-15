import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# utils
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from omegaconf import OmegaConf
from datetime import datetime

# Custom
from src.utils.Logger import ImagePredictionLogger

OmegaConf.register_new_resolver("now", lambda pattern: datetime.now().strftime(pattern))

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    wandb.login(key=cfg.wandb.key)

    dm = instantiate(cfg.data)
    early_stop_callback = instantiate(cfg.early_stopping)
    checkpoint_callback = instantiate(cfg.checkpoint)
    main_model = instantiate(cfg.model)

    dm.setup('fit')
    dm.setup('test')
    val_loader = dm.val_dataloader()

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
