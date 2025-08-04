# Rain Nowcasting with ConvLSTM/ConvRNN on SEVIR

Short-term precipitation nowcasting from satellite/radar imagery using recurrent
convolutional models (**ConvLSTM**, **ConvRNN**, **Peephole ConvLSTM**) with a
reproducible training pipeline (Hydra + PyTorch Lightning + W&B).

> **Data:** SEVIR, IR 6.9µm channel (ir069), sequences of 49 frames @ 5-min cadence, 192×192 px  
> **Stack:** PyTorch, Lightning, Hydra, Weights & Biases, HDF5 tooling

---

## At a glance

- **Goal:** predict the next frames (rain/cloud motion proxy) given a short context window.  
- **Scale:** ~45 GB of ir069 sequences in hierarchical HDF5 with CSV index; lazy loading for efficiency. :contentReference[oaicite:15]{index=15}  
- **Models:** ConvRNN, ConvLSTM, Peephole ConvLSTM; shared decoder; ablations with BN. :contentReference[oaicite:16]{index=16}  
- **Best (example):** ConvRNN, kernel 7, depth 1, channels 8 → **MSE ≈ 0.016** on test split. :contentReference[oaicite:17]{index=17}  
- **Caveat:** results show **instability** across identical configs → seeds/repro matter. :contentReference[oaicite:18]{index=18}

![sample_pred](docs/img/sample_prediction.png)
*Predicted vs ground-truth frames (placeholder).*

---

## Data

- **Source:** SEVIR (ir069 infrared channel), sequences of **49 frames** (4 hours), **192×192**.  
- **Access:** downloaded via `awscli`, stored as **10 HDF5** files with an index CSV.  
- **Splits:** storm events & mixed events; train/val/test created via index filters.  
- **IO:** custom `Dataset` with **lazy loading** to keep memory bounded; **Lightning DataModule** for reproducible splits and loaders. :contentReference[oaicite:19]{index=19}

![dataset_layout](docs/img/dataset_layout.png)
*Dataset structure & lazy loading pipeline (placeholder).*

---

## Model family

We evaluate several recurrent cells for spatiotemporal prediction:

- **ConvRNN** (lightweight)  
- **ConvLSTM** (Shi et al. 2015)  
- **Peephole ConvLSTM** (cell gains peephole connections)  

A common wrapper (**`RainPredictor`**) drives the cell over time; a shared **decoder** (conv layers) maps hidden states to the 1-channel output. Ablations include BatchNorm **inside** and **outside** the cell (BN inside RNN is generally ineffective). :contentReference[oaicite:20]{index=20}

![archs](docs/img/architectures.png)
*Cells and shared decoder (placeholder).*

---

## Training pipeline

- **Config:** **Hydra** hierarchical configs instantiate models/optim/schedules without boilerplate. :contentReference[oaicite:21]{index=21}  
- **Engine:** PyTorch Lightning (train/val/test loops, checkpoints).  
- **Logging:** **Weights & Biases** – losses, example predictions, configs.  
- **Hardware notes:** initial GCP T4 runs throttled by **disk IO (~160 MB/s)**; moving to local **NVMe (PCIe 5.0)** unlocked GPU utilization. :contentReference[oaicite:22]{index=22}

![training_curves](docs/img/training_curves.png)
*Training/validation loss and sample outputs (placeholder).*

---

## Results

**Representative scores (MSE↓)**

| Cell         | Kernel | Depth | Channels | MSE   |
|--------------|--------|-------|----------|-------|
| ConvRNN      | 7      | 1     | 8        | 0.016 |
| ConvLSTM     | 7      | 1     | 8        | 0.020 |
| Peephole LSTM| 7      | 1     | 12       | 0.025 |
| ConvRNN      | 5      | 3     | 8        | 0.026 |
| ConvLSTM     | 7      | 2     | 8        | 0.046 |

> Takeaways: with **short input contexts**, **ConvRNN** can outperform heavier ConvLSTM variants; Peephole + BN did not help on test. **Instability** observed: retraining the same config produced noticeably different curves → fix seeds, control data order, and report confidence intervals. :contentReference[oaicite:23]{index=23}

![instability](docs/img/instability.png)
*Two runs with identical config diverging in validation MSE (placeholder).*

---

## Repro: how to run

### 1) Environment
```bash
conda create -n raincast python=3.10 -y
conda activate raincast
pip install -r requirements.txt   # torch/lightning/hydra-core/wandb/h5py etc.
```

### 2) Data pointers

Create data/ with SEVIR h5 files and the index CSV. Update paths in configs/data/*.yaml

data/
  sevir/
    ir069/
      sevir_ir069_00.h5
      ...
    index.csv

### 3) Train

```
# Example: ConvRNN, kernel 7, depth 1, channels 8
python train.py \
  model=convrnn \
  model.kernel_size=7 model.depth=1 model.hidden_channels=8 \
  data=sevir_ir069 \
  trainer.max_epochs=30 \
  logger.wandb.project=raincast
```

### 3) Evaluate / visualize

```
python eval.py ckpt=path/to/checkpoint.ckpt
python viz_examples.py ckpt=... num_sequences=4 out=docs/img/samples.png
```
Hydra allows composing configs:

```
GSN_rain_predictor/
├─ configs/               # Hydra configs (model/data/trainer/experiment)
├─ data_utils/            # HDF5 readers, Dataset, DataModule
├─ models/                # ConvRNN, ConvLSTM, Peephole cells, decoder
├─ train.py               # Lightning Trainer bootstrap
├─ eval.py                # metrics on test
├─ viz_examples.py        # qualitative previews
├─ docs/
│  └─ img/                # put figures here (placeholders in this README)
└─ README.md
```

## Metrics & evaluation

- Primary: MSE on held-out sequences.
- Qualitative: frame grids, temporal plots per pixel location.
- Ablations: BN placement (inside vs outside the cell), decoder sharing, kernel size, channels, depth, batch size up to 4096 (stability for BN).

## Limitations

- Trained on a single channel (ir069) → limited cross-modal context.
- No uncertainty modeling (e.g., probabilistic forecasting).
- Result variance across runs; need strict seeding and deterministic dataloaders.
- No global post-processing (e.g., optical-flow constraints).

## Roadmap

- Add multi-channel inputs (vil, ir107) and simple fusion.
- Stabilize training (fixed seeds, deterministic loaders, CUDNN flags).
- Try transformers for Earth system forecasting (e.g., Earthformer) as a baseline.
- Metrics beyond MSE (e.g., SSIM) + probabilistic scores.

