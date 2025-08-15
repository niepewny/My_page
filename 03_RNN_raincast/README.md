# RNN Raincast - Short-term Precipitation Nowcasting on SEVIR (IR069)

> **Goal.** Forecast the next IR frames over short timespan from recent context frames using recurrent convolutional models (ConvRNN / ConvLSTM / Peephole ConvLSTM).

---

## TL;DR
- **Data:** SEVIR, infrared channel **IR069** (49 frames x 5-min, 192x192). We train on extracted sequences; a tiny sample is provided for quick demo.  
- **Models:** ConvRNN, ConvLSTM, Peephole-ConvLSTM with a lightweight decoding head; Hydra configs + PyTorch Lightning.  
- **Results:** Best validation/test **MSE ~ 0.016** (ConvRNN, kernel 7, depth 1, hidden 8). ConvLSTM / Peephole close behind in our setting.  
- **Findings:** BatchNorm inside RNN cells did **not** help; adding an external decoding head helped LSTM variants. Results can be **unstable** across seeds/hardware; we document this and suggest mitigations.  

**Full discussion of the project (in Polish) in ``docs/report.pdf``**.

---

## My role

This was a two-person project. My contributions:
- Design and implementation of the recurrent architectures (ConvRNN / ConvLSTM / Peephole ConvLSTM) and the decoding head.
- Implementation of auxiliary modules (configuration, utilities) and the end-to-end training loop.
- Experiment design and evaluation.
The report was created as a team.
**Note:** the `src/data_modules/` folder contains the code not authored by me.
---

## Project Structure

```
03_RNN_raincast/
|- src/                         
|   |- architectures/
|   |    |- conv_lstm.py
|   |    |- peephole_conv_lstm.py
|   |    |- conv_rnn.py
|   |- data_modules/            # data modules - not of my creation
|   |- predictors/
|   |    |-rain_predictor.py
|   |- utils/
|        |- logger.py
|- configs/                     # Hydra configs (model/data/trainer/experiment)
|   |- default.yml
|- notebooks/                   
|   |-demo.ipynb                # demo notebook
|- data/
|   |- sample_ir069_9f.npz      # tiny sample (for demo only)
|- checkpoints/                 # sample checkpoint
|- docs/
|   |- img/                     # figures used in README
|   |- report.pdf               # detailed report (PL)
|- train.py                     # training
|- requirements.txt
|- .gitignore
|- README.md                    # (this file)
```

---

## Modifiable architecture structure

```bash
Input: batch x T x C x H x W
        |
        v
+----------------------------------+
|          RainPredictor           |
|----------------------------------|
|  Backbone (loop over time):      |
|    - ConvRNN / ConvLSTM /        |
|      PeepholeConvLSTM            |
|       - parameters:              |
|           - input_channels       |
|           - hidden_channels      |
|           - kernel_size          |
|           - depth                |
|           - activation           |
|----------------------------------|
|  Mapping Layer:                  |
|    - BatchNorm                   |
|    - Conv2d (kernel_size)        |
|    - BatchNorm                   |
|    - Conv2d (1x1)                |
|    - Activation                  |
|----------------------------------|
+----------------------------------+
        |
        v
Output: batch x 1 x H x W

```

---

## Data (SEVIR, IR069)

We use the **SEVIR** dataset and focus on the **IR069** infrared channel for nowcasting.  
Each sequence has **49 frames** at **5-minute** intervals (**4 hours** total), with **192×192** spatial resolution. We access the public AWS copy and keep indices in a CSV; the full IR069 subset is ~**45 GB** across **10** HDF5 files.  
For convenience, we include a *tiny* demo sample under `data/sample_ir069_9f.npz` (9 frames). Please use the official source for full training.

**Get the data (recommended):**
- Follow SEVIR instructions (AWS CLI) to download the IR069 HDF5s.  
- Keep paths configurable via Hydra (see `configs/data/*.yaml`).  
- Do **not** commit large HDF5 files to the repo; use a local path or LFS/releases if needed.

---

## Setup
### Requirements
 - AWS
 - CUDA (if training on GPU)

```bash
# Python 3.10+ recommended
python -m venv .venv 
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
Install PyTorch with CUDA following the official instructions for your CUDA/driver version.
---

## Quickstart

### 1) Notebook demo (no big downloads)
Open the demo notebook and run all cells:
```bash
jupyter lab notebooks/demo.ipynb
```
The notebook loads `data/sample_ir069_9f.npz` and runs a forward pass with a small ConvLSTM/ConvRNN model, plotting input vs predicted frames.

### 2) Train & evaluate (Hydra + Lightning)
```bash
# Example: ConvLSTM
python train.py model.type=convlstm model.hidden_channels=8 model.kernel_size=7 model.depth=1 \
               data.sequence_len=49 trainer.max_epochs=30

# Evaluate a checkpoint (adjust path)
python train.py mode=eval ckpt_path=checkpoints/sample.pt
```

**Tips**
- Set `data.step` to subsample input frames (e.g., `step=2` to take every 2nd frame).  
- Use `trainer.accelerator='gpu'` and `trainer.devices=1` when a GPU is available.  
- All hyperparameters are overridable from the CLI thanks to Hydra.

---

## Models

We implement three recurrent cells for spatiotemporal modeling:
- **ConvRNN** — convolutional RNN cell (fast/simple baseline).  
- **ConvLSTM** — standard ConvLSTM cell.  
- **Peephole ConvLSTM** — ConvLSTM variant with peephole connections.

A lightweight **decoding head** (conv layers) maps the hidden state to a single-channel prediction. We found that moving the decoder outside the cell helps LSTM variants.

---

## DataModule & Transforms

We use a PyTorch Lightning **DataModule** to:  
- split sequences into **train/val/test**,  
- lazily load only the needed sequence for `__getitem__`,  
- apply transforms (resize to `height×width`, step, cropping, normalization).

Tune these common knobs:
- `sequence_len` (default 49), `step` (subsampling stride),  
- `height`, `width`, `batch_size`, `num_workers`,  
- `train/val/test percents` for splitting.

---

## Results (examples)

> Numbers below are representative examples from our experiments; use the notebook and training script to reproduce on your setup.

| RNN Type | Kernel | Depth | Hidden | **MSE** |
|:--------:|:------:|:-----:|:------:|--------:|
| ConvRNN  |   7    |   1   |   8    | **0.016** |
| ConvLSTM |   7    |   1   |   8    | 0.020 |
| Peephole ConvLSTM | 7 | 1 | 12 | 0.025 |
| ConvRNN  |   5    |   1   |  12    | 0.018 |
| ConvLSTM |   3    |   2   |  12    | 0.034 |

> We also observed that **BatchNorm inside the RNN cell did not improve** validation/test performance. Placing BN outside the cell helped stabilize learning for LSTM variants in our setting.

---

## Limitations & Next Steps

- **Single channel (IR069) only.** Extend to multi-channel training/fusion.  
- **Result stability.** Aggregate over **multiple seeds**; try curriculum data or stronger regularization.  
- **Larger context.** Test longer sequences and downsampled inputs (trade-off between temporal context and compute).  
- **Earthformer/Transformers.** Compare against modern spatiotemporal transformers as resources allow.

## Configuration

The project uses Hydra. All used configurations are accessible via configs/dafault.yaml

### Choosing the type of RNN cell

The parameter in the configuration - ``model.RNN_cell._target_``
- Options to choose:
    - ``src.architectures.conv_rnn.ConvRNNCell``
    - ``src.architectures.conv_lstm.ConvLSTMCell``
    - ``src.architectures.peephole_conv_lstm.ConvPeepholeLSTMCell``