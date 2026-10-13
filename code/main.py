#!/usr/bin/env python

import os
import numpy as np
from scaling import MAX_SLIP, MAX_RISE, MAX_RUPT

def main():
    """Train the Longmenshan STGNN model."""
    # Set the CUDA allocator before importing the trainer and PyTorch.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    from trainer import train

    test_num = '01'
    params = {
        'Testnum': test_num,
        'epochs': 350,
        'batch_size': 64,
        'lr': 1e-3,
        'out_dir': f"./checkpoints/Test{test_num}",
        'Tlen': 512,
        'Nstan': 60,
        'Nfault': 985,
        'dt_wave': 0.5,
        'max_vals': {
            'slip': MAX_SLIP,
            'rise': MAX_RISE,
            'rupt': MAX_RUPT,
        },
    }

    files = {
        'GFlist': '/root/autodl-tmp/station.txt',
        'GF_tensor': './Split_info/Kobayashi_GF_4D.npy',
    }

    gf_path = files['GF_tensor']
    if not os.path.exists(gf_path):
        raise FileNotFoundError(f"Green's function tensor not found: {gf_path}")

    print(f"Loading Green's function tensor: {gf_path}")
    gf_tensor = np.load(gf_path)
    print(f"Tensor shape: {gf_tensor.shape}")

    os.environ["OMP_NUM_THREADS"] = "8"
    train(files, params, gf_tensor)

    checkpoint = os.path.join(params['out_dir'], 'best_model.pth')
    print(f"Training complete. Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
