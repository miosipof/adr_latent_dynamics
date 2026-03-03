# node_train.py

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import h5py
import numpy as np
import json
import copy
import torch
import argparse

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.dataset import H5Snapshots, get_dataset, H5WindowedTrajectories, load_time_segment_from_h5
from src.utils import seed_everything

from src.geometry import (
    iso_penalty_exact,
    iso_penalty_stochastic,
    operator_norm_loss,
    curvature_penalty
)

from src.node import (
    AffineModulatedLatentODEFunc, 
    train_latent_dynamics, 
    make_seeded_loader, 
    get_shared_init_from_vanilla
)

from src.autoencoder import (
    SpatialCoherentEncoder, 
    SpatialCoherentDecoder, 
    infer_L_for_latent, 
    stiefel_project_conv_columns,
    eval_ae
)


# Constants ----> YAML
EPOCHS = 50
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
latent_H = 4  # 4x4 => n=16
B = 64
L = infer_L_for_latent(H=32, latent_H=latent_H)

path = "adr_full.h5"

L_max = 40
mu_dim = 3  # ADR params dimension
REG = 0.1   # Regularization strength


experiments = {
    "vanilla": {
        "lam_reg": 0,
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
        "lam_reg": 0,
        "reg_fn": None
    }
}


def make_train_loader(ds, B, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=B, shuffle=True, generator=g, num_workers=0, pin_memory=True)

def make_init_state(ode_cls, ode_kwargs, init_seed: int, device="cuda"):
    """
    Creates ONE random initialization for ode_cls(**ode_kwargs),
    and returns a CPU state_dict that you can load into every method's NODE.
    """
    # Keep init reproducible
    cuda_devs = [torch.cuda.current_device()] if (device.startswith("cuda") and torch.cuda.is_available()) else []
    with torch.random.fork_rng(devices=cuda_devs):
        torch.manual_seed(init_seed)
        if cuda_devs:
            torch.cuda.manual_seed_all(init_seed)

        ode0 = ode_cls(**ode_kwargs).to("cpu")
        return copy.deepcopy(ode0.state_dict())

        
    
def main(V=1, run=1):

    seed = 100*V + run
    seed_everything(seed=seed, deterministic=True)
    
    mu_train, train_idx, val_idx, t_coarse, t_fine, coord_grid, Nmu, Nt_coarse, H, W = get_dataset(path) 

    T_max = float(t_coarse[-1])

    ds_train = H5Snapshots(path, split="train", field="u_coarse", mu_indices=train_idx, return_mu=False)

    t0_tr, t1_tr = load_time_segment_from_h5(path, kind="coarse", segment="train")
    t0_va, t1_va = load_time_segment_from_h5(path, kind="coarse", segment="val_loop")
    train_t_ids = np.arange(t0_tr, t1_tr + 1, dtype=np.int64)
    
    ds_dyn_train = H5WindowedTrajectories(path, split="train", field="u_coarse",
                                          mu_indices=train_idx, lmax=L_max,
                                          t_start=t0_tr, t_end=t1_tr, return_mu=True)
    ds_dyn_val   = H5WindowedTrajectories(path, split="train", field="u_coarse",
                                          mu_indices=val_idx, lmax=L_max,
                                          t_start=t0_va, t_end=t1_va, return_mu=True)

    print(f"Datasets ready: train={ds_dyn_train.__len__()}, val={ds_dyn_val.__len__()}")
    

    
    umin, umax = ds_train.compute_train_minmax(
        field="u_coarse",
        mu_ids=train_idx,
        t_ids=train_t_ids,
        split="train",
    )
    
    ds_dyn_train.umin, ds_dyn_train.umax = umin, umax
    ds_dyn_val.umin,   ds_dyn_val.umax   = umin, umax

    


    ode_kwargs = dict(nmu=mu_dim, k=8, Tmax=T_max, de=64, nc=32, z_channels=1)

    # shared random init for this (V, run)
    init_seed = 10000 + seed
    init_state = make_init_state(
        AffineModulatedLatentODEFunc, ode_kwargs, init_seed=init_seed, device=DEVICE
    )   

    
    for exp_name, exp_setting in experiments.items():

        lam_reg = exp_setting["lam_reg"]
        reg_fn = exp_setting["reg_fn"]

        ode_func = AffineModulatedLatentODEFunc(**ode_kwargs)

        dl_dyn_train = make_seeded_loader(ds_dyn_train, batch_size=B, shuffle=True,  seed=seed, num_workers=0)
        dl_dyn_val   = make_seeded_loader(ds_dyn_val,   batch_size=B, shuffle=False, seed=seed, num_workers=0)

        stiefel_project = stiefel_project_conv_columns if exp_name == "stiefel" else None
        # stiefel_project = None
        
        enc = torch.load(f"output/AE_v{V}/{exp_name}/enc_target.pt", map_location=DEVICE, weights_only=False)
        dec = torch.load(f"output/AE_v{V}/{exp_name}/dec_target.pt", map_location=DEVICE, weights_only=False)
        
        ode_func, losses = train_latent_dynamics(
            enc, dec, ode_func,
            dl_dyn_train, 
            dl_dyn_val,
            epochs=EPOCHS,
            init_ode_state_dict=init_state,
            device=DEVICE,
            lr=LR,
            schedule="exponential",
            exp_gamma=0.95,
            warmup=0.1,
            grad_clip=1.0,
            curriculum=0.0, # NO CURRICULUM for horizon: use full L_max
            deterministic_ell=True,
            stiefel_project_fn=stiefel_project,
            seed=seed,
            lmax=L_max,
            rk_method="rk2",
            exp_name=exp_name,
            V=V,
            run=run,
            train_ae=False, # Unfreeze Autoencoder?
            reg_fn=reg_fn,
            lam_reg=lam_reg,
            lam_recon=1.0,
            w_anchor=1.0
        )






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NODE training")
    
    parser.add_argument('--V', type=int, required=True,
                        help='AE version = 1, 2, ...')

    parser.add_argument('--run', type=int, required=True,
                        help='Run = 1, 2, ...')    

    args = parser.parse_args()

    main(args.V, args.run)