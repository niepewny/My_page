from pytorch_lightning.callbacks import Callback
import wandb
import numpy as np


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=1):
        super().__init__()

        self.x = val_samples[:, :-1]
        self.y = val_samples[:, -1]
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, model):
        x = self.x.to(device=model.device)
        outputs = model(x)

        y = self.y[:self.num_samples].cpu().numpy()
        outputs = outputs[:self.num_samples].cpu().detach().numpy()

        logged_images = []
        for true_img, pred_img in zip(y, outputs):
            combined_img = np.concatenate([true_img, pred_img], axis=2)
            normalized_img = (combined_img - combined_img.min()) / (combined_img.max() - combined_img.min())
            logged_images.append(
                wandb.Image(normalized_img, caption="Ground Truth | Prediction")
            )

        trainer.logger.experiment.log({
            "examples": logged_images
        })
