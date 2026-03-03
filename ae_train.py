# ae_train.py

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import h5py
import numpy as np
import json
import torch
import argparse

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.dataset import H5Snapshots, get_dataset, load_time_segment_from_h5
from src.utils import seed_everything
from src.autoencoder import (
    SpatialCoherentEncoder, 
    SpatialCoherentDecoder, 
    infer_L_for_latent, 
    eval_ae,
    train_ae
)
from src.geometry import (
    iso_penalty_exact,
    iso_penalty_stochastic,
    operator_norm_loss,
    curvature_penalty
)


# Constants ----> YAML
EPOCHS = 200
LR = 5e-4
MSE_TARGET = 1.0e-5
WARMUP = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
latent_H = 4  # 4x4 => n=16
B = 128
L = infer_L_for_latent(H=32, latent_H=latent_H)
path = "adr_full.h5"

REG = 0.1

experiments = {
    "vanilla": {
        "lam_reg": 0.0,
        "reg_fn": None
    },
    "stoch_iso": {
        "lam_reg": REG,
        "reg_fn": iso_penalty_stochastic
    },    
    "operator_norm": {
        "lam_reg": REG,
        "reg_fn": operator_norm_loss
    },
    "curvature": {
        "lam_reg": REG*10,
        "reg_fn": curvature_penalty
    },
    "stiefel": {
        "lam_reg": 0.0,
        "reg_fn": None
    }
}



def main(exp_name, V=1):

    seed_everything(seed=V, deterministic=True)

    mu_train, train_idx, val_idx, t_coarse, t_fine, coord_grid, Nmu, Nt_coarse, H, W = get_dataset(path)

    t0_tr, t1_tr = load_time_segment_from_h5(path, kind="coarse", segment="train")
    t0_va, t1_va = load_time_segment_from_h5(path, kind="coarse", segment="val_loop")
    train_t_ids = np.arange(t0_tr, t1_tr + 1, dtype=np.int64)

    # AE train/val snapshot loaders using the parameter split indices
    ds_train = H5Snapshots(path, split="train", field="u_coarse", mu_indices=train_idx, t_indices=train_t_ids, return_mu=False)
    ds_val   = H5Snapshots(path, split="train", field="u_coarse", mu_indices=val_idx, t_indices=train_t_ids, return_mu=False)
    
    umin, umax = ds_train.compute_train_minmax(
        field="u_coarse",
        mu_ids=train_idx,
        t_ids=train_t_ids,
        split="train",
    )
    
    ds_train.umin, ds_train.umax = umin, umax
    ds_val.umin, ds_val.umax = umin, umax

    
    # # [DEBUG]
    # ds_train = torch.utils.data.Subset(ds_train, range(10))
    # ds_val = torch.utils.data.Subset(ds_val, range(2))
    # B = 2

    print(f"Dataset loaded: {len(ds_train)} train | {len(ds_val)} val")

    g = torch.Generator()
    g.manual_seed(V)
    
    dl_train = DataLoader(ds_train, batch_size=B, shuffle=True, generator=g, num_workers=0)
    dl_val   = DataLoader(ds_val, batch_size=B, shuffle=False, num_workers=0)


    # for exp_name, exp_setting in experiments.items():

    exp_setting = experiments[exp_name]

    
        
    lam_reg = exp_setting["lam_reg"]
    reg_fn  = exp_setting["reg_fn"]

    print(f"Starting {exp_name} AE training [V {V}], lam_reg={lam_reg}")
        
    enc, dec, losses = train_ae(dl_train, dl_val, lr=LR, L=L, 
                 lam_reg=lam_reg, mse_target=MSE_TARGET, reg_fn=reg_fn, warmup=WARMUP, schedule="exponential",
                 exp_name=exp_name, V=V, device=DEVICE, epochs=EPOCHS)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AE training")
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='vanilla, stoch_iso, operator_norm, curvature, stiefel')
    
    parser.add_argument('--V', type=int, required=True,
                        help='Version = 1, 2, ...')

    args = parser.parse_args()

    main(args.exp_name, args.V)
    