#!/usr/bin/env python
"""
STGNN training with waveform augmentation and loss history export.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import PhysicsInformedSTGNN
from scaling import d_scale, r_scale, t_scale, m_scale
from loss import compute_loss

# Enable TensorFloat-32 on supported GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SeismicDataset(Dataset):
    def __init__(self, E_paths, N_paths, Z_paths, y_dict, adj_matrix, noise_bank=None, is_train=False, Tlen=512,
                 Nstan=50):
        self.E_paths = E_paths
        self.N_paths = N_paths
        self.Z_paths = Z_paths
        self.y = y_dict
        self.adj = adj_matrix
        self.noise_bank = noise_bank
        self.is_train = is_train
        self.Tlen = Tlen
        self.Nstan = Nstan

    def __len__(self):
        return len(self.E_paths)

    def __getitem__(self, idx):
        # Load clean three-component waveforms.
        path_E = str(self.E_paths[idx])
        path_N = str(self.N_paths[idx])
        path_Z = str(self.Z_paths[idx])

        wave_E = np.load(path_E, mmap_mode='r')[:self.Nstan, :self.Tlen]
        wave_N = np.load(path_N, mmap_mode='r')[:self.Nstan, :self.Tlen]
        wave_Z = np.load(path_Z, mmap_mode='r')[:self.Nstan, :self.Tlen]

        x_wave_clean = np.stack([wave_E, wave_N, wave_Z], axis=0).astype(np.float32, copy=True)
        x_wave_input = x_wave_clean.copy()

        keep_mask = np.ones(self.Nstan, dtype=bool)

        # Augment the input while retaining clean waveform targets.
        if self.is_train:
            # Randomly drop stations.
            drop_prob = np.random.uniform(0.0, 0.2)
            keep_mask = np.random.rand(self.Nstan) > drop_prob
            x_wave_input[:, ~keep_mask, :] = 0.0

        # Add sampled red noise.
        if self.noise_bank is not None:
            num_noise = self.noise_bank.shape[0]
            idx_n = np.random.randint(0, num_noise, size=(3, self.Nstan))

            sampled_noise = self.noise_bank[idx_n, :self.Tlen].astype(np.float32, copy=False)
            max_amp = np.max(np.abs(x_wave_input), axis=2, keepdims=True)

            snr = np.random.uniform(10.0, 50.0, size=(3, self.Nstan, 1)).astype(np.float32)
            valid_mask = max_amp > 1e-10

            scaled_noise = np.zeros_like(x_wave_input, dtype=np.float32)
            valid_2d = valid_mask.squeeze(-1)
            scaled_noise[valid_2d] = (sampled_noise[valid_2d] * (max_amp[valid_2d] / snr[valid_2d]))

            x_wave_input += scaled_noise

        # Normalize source and magnitude targets.
        ss_raw = self.y['ss_slip'][idx]
        ds_raw = self.y['ds_slip'][idx]
        rise_raw = self.y['rise_time'][idx]
        rupt_raw = self.y['rupt_time'][idx]
        mag_raw = self.y['mag_output'][idx]

        ss_norm = d_scale(ss_raw)
        ds_norm = d_scale(ds_raw)
        rise_norm = r_scale(rise_raw)

        y_source = np.stack([ss_norm, ds_norm, rise_norm], axis=0).astype(np.float32)
        y_rupt = np.asarray(t_scale(rupt_raw), dtype=np.float32)
        y_mag = (
            float(mag_raw[0])
            if isinstance(mag_raw, (np.ndarray, list))
            else float(mag_raw)
        )

        y_mag_norm = float(m_scale(y_mag))

        return {
            'input_wave': torch.from_numpy(x_wave_input),
            'input_adj': torch.as_tensor(self.adj, dtype=torch.float32),
            'targets': {
                'source_output': torch.from_numpy(y_source),
                'rupt_output': torch.from_numpy(y_rupt),
                'mag_output': torch.tensor([y_mag_norm], dtype=torch.float32),
                'waveform_output': torch.from_numpy(x_wave_clean)
            }
        }


def run_epoch(model, loader, optimizer, device, phase='S1', is_train=True, fault_adj=None):
    if is_train:
        model.train()
    else:
        model.eval()

    metrics = {'total': 0, 'src': 0, 'rup': 0, 'wave': 0, 'mag': 0, 'sparse': 0}
    num_batches = len(loader)

    pbar = tqdm(loader, desc="Train" if is_train else "Valid", leave=False)

    for batch in pbar:
        wave = batch['input_wave'].to(device, non_blocking=True)
        adj = batch['input_adj'].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch['targets'].items()}

        if is_train: optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(wave, adj, fault_adj=fault_adj)
            loss, l_src, l_rup, l_wave, l_mag, l_sparse = compute_loss(targets, preds, phase=phase)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics['total'] += loss.item()
        metrics['src'] += l_src.item()
        metrics['rup'] += l_rup.item()
        metrics['wave'] += l_wave.item()
        metrics['mag'] += l_mag.item()
        metrics['sparse'] += l_sparse.item()

    for k in metrics: metrics[k] /= num_batches
    return metrics


def train(files, params, gf_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DATA_ROOT = './Split_info'

    if not os.path.exists(os.path.join(DATA_ROOT, 'X_train_E.npy')):
        raise FileNotFoundError(f"Missing training data: {DATA_ROOT}/X_train_E.npy")

    n_epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 16)
    lr = params.get('lr', 1e-3)
    out_dir = params.get('out_dir', './Runs/GAT_Test')

    if not os.path.exists(out_dir): os.makedirs(out_dir)

    E_train = np.load(os.path.join(DATA_ROOT, 'X_train_E.npy'))
    N_train = np.load(os.path.join(DATA_ROOT, 'X_train_N.npy'))
    Z_train = np.load(os.path.join(DATA_ROOT, 'X_train_Z.npy'))
    y_train = np.load(os.path.join(DATA_ROOT, 'y_train.npy'), allow_pickle=True).item()

    E_valid = np.load(os.path.join(DATA_ROOT, 'X_valid_E.npy'))
    N_valid = np.load(os.path.join(DATA_ROOT, 'X_valid_N.npy'))
    Z_valid = np.load(os.path.join(DATA_ROOT, 'X_valid_Z.npy'))
    y_valid = np.load(os.path.join(DATA_ROOT, 'y_valid.npy'), allow_pickle=True).item()

    adj_matrix_path = os.path.join(DATA_ROOT, 'adj_matrix.npy')
    adj_matrix_np = np.load(adj_matrix_path)

    fault_adj_path = os.path.join(DATA_ROOT, 'fault_adj.npy')
    fault_adj_matrix = np.load(fault_adj_path)
    fault_adj_tensor = torch.tensor(fault_adj_matrix, dtype=torch.float32).to(device)

    noise_path = os.path.join(DATA_ROOT, 'noise_bank.npy')
    if os.path.exists(noise_path):
        print("Noise augmentation: enabled")
        noise_bank_np = np.load(noise_path)
    else:
        print("Noise augmentation: disabled")
        noise_bank_np = None

    print(f"Samples: {len(E_train)} train, {len(E_valid)} validation")

    n_stan = params.get('Nstan', 60)

    train_dataset = SeismicDataset(E_train, N_train, Z_train, y_train, adj_matrix_np,
                                   noise_bank=noise_bank_np, is_train=True, Nstan=n_stan)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    val_dataset = SeismicDataset(E_valid, N_valid, Z_valid, y_valid, adj_matrix_np,
                                 noise_bank=None, is_train=False, Nstan=n_stan)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = PhysicsInformedSTGNN(params, gf_tensor).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    writer = SummaryWriter(os.path.join(out_dir, 'logs'))

    best_val_loss = float('inf')
    early_stop_patience = 50
    early_stop_counter = 0

    history_records = []

    for epoch in range(1, n_epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, phase='S1',
                            is_train=True, fault_adj=fault_adj_tensor)

        val_m = run_epoch(model, val_loader, optimizer, device, phase='S1',
                          is_train=False, fault_adj=fault_adj_tensor)

        scheduler.step(val_m['total'])
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalars('Loss/0_Total', {'train': train_m['total'], 'val': val_m['total']}, epoch)
        writer.add_scalars('Loss/1_Source(Slip_Rise)', {'train': train_m['src'], 'val': val_m['src']}, epoch)
        writer.add_scalars('Loss/2_RuptureTime', {'train': train_m['rup'], 'val': val_m['rup']}, epoch)
        writer.add_scalars('Loss/3_Waveform', {'train': train_m['wave'], 'val': val_m['wave']}, epoch)
        writer.add_scalars('Loss/4_Magnitude', {'train': train_m['mag'], 'val': val_m['mag']}, epoch)
        writer.add_scalars('Loss/5_Sparsity', {'train': train_m['sparse'], 'val': val_m['sparse']}, epoch)

        print(
            f"Epoch {epoch:03d}/{n_epochs} | LR {current_lr:.1e} | "
            f"train {train_m['total']:.4f} | val {val_m['total']:.4f}"
        )

        history_records.append({
            'Epoch': epoch,
            'LR': current_lr,
            'Train_Total_Loss': train_m['total'],
            'Val_Total_Loss': val_m['total'],
            'Train_Src_Loss': train_m['src'],
            'Val_Src_Loss': val_m['src'],
            'Train_Rup_Loss': train_m['rup'],
            'Val_Rup_Loss': val_m['rup'],
            'Train_Wave_Loss': train_m['wave'],
            'Val_Wave_Loss': val_m['wave'],
            'Train_Mag_Loss': train_m['mag'],
            'Val_Mag_Loss': val_m['mag'],
            'Train_Sparse_Loss': train_m['sparse'],
            'Val_Sparse_Loss': val_m['sparse']
        })

        history_csv_path = os.path.join(out_dir, 'training_history.csv')
        pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

        if val_m['total'] < best_val_loss:
            best_val_loss = val_m['total']
            save_path = os.path.join(out_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping: no validation improvement for {early_stop_patience} epochs.")
            break

    writer.close()
    return {'best_val_loss': best_val_loss}


if __name__ == "__main__":
    print("Run main.py to start training.")
