# PI-FRONT model weights

The checkpoint is provided as a GitHub Release asset, not as a file in the Git repository.

## Download

[Download best_model.pth](https://github.com/Li7654321/FRONT-and-PI-FRONT/releases/tag/v1.0.0/best_model.pth)


## Training configuration

| Parameter | Value |
| --- | --- |
| Python / PyTorch / CUDA | 3.12 / 2.7.0+cu128 / 12.8 |
| Maximum epochs | 350 |
| Batch size | 64 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Learning-rate scheduler | ReduceLROnPlateau; factor 0.5, patience 10 |
| Early stopping patience | 50 epochs |
| Stations / components / samples | 60 / 3 / 512 |
| Sampling interval | 0.5 s |
| Fault subfaults | 985 |
| Data split | 70% training / 20% validation / 10% test |


## Loss weights

The manuscript describes the objective as:

| Loss term | FRONT weight |
| --- | --- |
| Source parameters | 1.0 |
| Rupture time | 2.0 |
| Magnitude | 1.0 |
| Slip sparsity | 0.5 |
| Waveform | 1.5 |
