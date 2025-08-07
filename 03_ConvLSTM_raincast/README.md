# GSN_rain_predictor

## Pobieranie SEVIR dataset
tylko częscti czyli ir069
```
Data key: ir107
Description: Infrared Satellite imagery (Window)
Spatial Resolution: 2 km
Patch Size: 192 x 192
Time step: 5 minutes
```
w folderze projektowym wykonać komendę:

mkdir dataset
cd dataset
aws s3 sync --no-sign-request s3://sevir/data/ir069 data/

## Trenowanie modelu
W pliku ``config.yaml`` należy wskazać ścieżkę do datasetu - ``data.dir``, klucz do Wandb - ``wandb.key``, oraz o ile jest to konieczne moduł obliczeniowy (gpu/cpu) - ``trainer.accelerator``, ``trainer.devices``. Edycja pozostałych parametrów jest dowolna.

Zmiana używanego modelu:
``model.RNN_cell._target_:`` *[do wyboru]*

*src.architectures.ConvRNN.ConvRNNCell* 

*src.architectures.ConvLSTM.ConvLSTMCell* 

*src.architectures.PeepholeConvLSTM.ConvPeepholeLSTMCell* 

(dla poprawnego nazewnictwa checkpointów) ``model.RNN_cell_name``
