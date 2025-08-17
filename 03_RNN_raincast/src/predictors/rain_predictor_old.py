import pytorch_lightning as pl
import torch
import torch.nn as nn


class RainPredictor(pl.LightningModule):

    def __init__(self, model, mapping_activation, mapping_kernel_size, learning_rate, loss_metrics, scheduler_step, scheduler_gamma):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model
        self.mapping_activation = mapping_activation
        self.lr = learning_rate
        self.loss = loss_metrics
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma

        hidden_channels = model.hidden_channels

        self.mapping_layer = nn.Sequential(
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=mapping_kernel_size,
                padding=mapping_kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=1,
                padding=0,
                bias=False
            ),
        )      
          
        self.current_epoch_training_loss = torch.tensor(0.0)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
        self.model.initialize_hidden_state(batch_size, height, width, x.device)
        for i in range(sequence_length):
            outputs = self.model(x[:, i], gen_output=(i == sequence_length-1))
        
        outputs = self.mapping_layer(outputs)
        outputs = self.mapping_activation(outputs)

        return outputs

    def compute_loss(self, y_pred, y):
        #TODO - move to data module
        y = y.contiguous()
        return self.loss(y_pred, y)

    def common_step(self, batch):
        x = batch[:, :-1]
        y = batch[:, -1]
        outputs = self(x)
        loss = self.compute_loss(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch):
        loss, outputs, y = self.common_step(batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch)
        self.training_step_outputs.append(loss)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        return {'loss': loss}

    def on_train_epoch_end(self):
        outs = torch.stack(self.training_step_outputs)
        self.current_epoch_training_loss = outs.mean()
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch)
        self.validation_step_outputs.append(loss)
        self.log_dict(
            {
                "validation_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return {'validation_loss': loss}

    def on_validation_epoch_end(self):
        outs = torch.stack(self.validation_step_outputs)
        avg_loss = outs.mean()

        self.log('validation_loss', avg_loss, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch)
        self.validation_step_outputs.append(loss)
        self.test_step_outputs.append(loss)
        self.log_dict(
            {
                "test_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        outs = torch.stack(self.test_step_outputs)
        avg_loss = outs.mean()
        self.log('test_loss_epoch', avg_loss, on_epoch=True)
        self.test_step_outputs.clear()

    # We should decide if we should stick to Adam/scheduler or dump it to hydra
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)
        return [optimizer], [lr_scheduler]