from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import torch
from SEVIR_data_loader import SEVIR_dataset
from torch.utils.data import Dataset
from vizualization import visualize_tensor_interactive
import hydra


###############
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.predictors.RainPredictor import RainPredictor
from src.data_modules.SEVIR_data_loader import ConvLSTMSevirDataModule
import torch

from omegaconf import OmegaConf

def load_model(checkpoint_path: str, config_path: str) -> RainPredictor:
    cfg = OmegaConf.load(config_path)

    # model z checkpointu
    model = RainPredictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        mapping_kernel_size=cfg.model.mapping_kernel_size,
        model=hydra.utils.instantiate(cfg.model.model),
        mapping_activation=hydra.utils.instantiate(cfg.model.mapping_activation),
        learning_rate=cfg.model.learning_rate,
        loss_metrics=hydra.utils.instantiate(cfg.model.loss_metrics),
        scheduler_step=cfg.model.scheduler_step,
        scheduler_gamma=cfg.model.scheduler_gamma
    )

    model.eval()  # Ustaw model w tryb ewaluacji
    return model

##############


if __name__ == "__main__":
    
    file_path_h5_dir = "D:\\gsn_dataset\\all\\2018"
    index = 0
    # przyjmuje step oraz szerokość i wysokość obrazka, oraz długość sekwencji(ucinamy 2 klatki tutaj)
    full_dataset = SEVIR_dataset(file_path_h5_dir, 6, 64, 64, 8)
    tensor = full_dataset.__getitem__(index)
    print(tensor.shape)
    
    tensor = tensor.squeeze(1)
    tensor = tensor.permute(1, 2, 0)
    
    visualize_tensor_interactive(tensor, "frame")
    
    ########
    
    model_weights_path = "model-epoch=14-validation_loss=0.02.ckpt"
    # yaml_config_hydra = "/home/kolaj/my_project/AI/PyTorch/GSN_rain_predictor/outputs/Cloud-ConvLSTM/wandb/run-20250105_225907-j97g5jg5/files/config.yaml"
    yaml_config_hydra = "config.yaml"
    # dm = ConvLSTMSevirDataModule(
    #     step=6,
    #     width=64,
    #     height=64,
    #     batch_size=1,
    #     num_workers=1,
    #     sequence_length=8,
    #     train_files_percent=0.7,
    #     val_files_percent=0.15,
    #     test_files_percent=0.15,
    #     files_dir=file_path_h5_dir)

    # dm.setup('test')
    # test_loader = dm.test_dataloader()
    # try:
    #     batch = next(iter(test_loader))
    # except exception as e:
    #     print("No data in test_loader")
    #     sys.exit(1)

    pre_trained_model = load_model(model_weights_path, yaml_config_hydra)

    with torch.no_grad():
        outputs = pre_trained_model(tensor)

    print(outputs.shape)