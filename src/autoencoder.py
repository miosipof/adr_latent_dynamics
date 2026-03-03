# src/autoencoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import shutil
from torch.optim import Adam
import os
import json



class PreActResBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.act = nn.ELU()
        self.conv_in  = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, bias=True)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=pad, bias=True)

    def forward(self, x):
        y = self.act(x)
        y = self.conv_in(y)
        y = self.act(y)
        y = self.conv_out(y)
        return x + y

class DownStage(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.res = PreActResBlock2d(channels)

    def forward(self, x):
        x = self.res(x)
        return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

class UpStage(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.res = PreActResBlock2d(channels)

    def forward(self, x):
        x = self.res(x)
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

def infer_L_for_latent(H, latent_H):
    ratio = H // latent_H
    if H % latent_H != 0:
        raise ValueError(f"H={H} must be divisible by latent_H={latent_H}")
    L = int(round(math.log2(ratio)))
    if 2**L != ratio:
        raise ValueError(f"H/latent_H={ratio} is not a power of 2.")
    return L

class SpatialCoherentEncoder(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=32, latent_ch=1, L=3, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_in  = nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=pad, bias=True)
        self.downs = nn.ModuleList([DownStage(hidden_ch) for _ in range(L)])
        self.conv_out = nn.Conv2d(hidden_ch, latent_ch, kernel_size=kernel_size, padding=pad, bias=True)

    def forward(self, u):
        x = self.conv_in(u)
        for d in self.downs:
            x = d(x)
        return self.conv_out(x)

class SpatialCoherentDecoder(nn.Module):
    def __init__(self, out_ch=1, hidden_ch=32, latent_ch=1, L=3, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_in  = nn.Conv2d(latent_ch, hidden_ch, kernel_size=kernel_size, padding=pad, bias=True)
        self.ups = nn.ModuleList([UpStage(hidden_ch) for _ in range(L)])
        self.conv_out = nn.Conv2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=True)

    def forward(self, z):
        x = self.conv_in(z)
        for u in self.ups:
            x = u(x)
        return self.conv_out(x)


@torch.no_grad()
def eval_ae(dl, enc, dec, device="cpu"):
    enc.eval(); dec.eval()
    tot = 0.0; n = 0
    for x in dl:
        x = x.to(device)
        z = enc(x)
        xhat = dec(z)
        loss = F.mse_loss(xhat, x, reduction="sum")
        tot += float(loss.cpu())
        n += x.numel()
    return tot / n  # per-pixel MSE


def train_ae(dl_train, dl_val, lr, L=3, 
             lam_reg=1e-2, mse_target=None, reg_fn=None, warmup=0.5, schedule="exponential",
             exp_name="vanilla", V=1, device="cpu", epochs=10, w_anchor=1.0):
    
    enc = SpatialCoherentEncoder(in_ch=1, hidden_ch=32, latent_ch=1, L=L).to(device)
    dec = SpatialCoherentDecoder(out_ch=1, hidden_ch=32, latent_ch=1, L=L).to(device)
    opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-5)

    if schedule=="exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    elif schedule=="plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    save_dir = f"output/AE_v{V}/{exp_name}"
    print(f"Working dir: {save_dir}")
    try:
        shutil.rmtree(save_dir);
    except:
        pass
        
    os.makedirs(save_dir, exist_ok=True)
    losses = {
        "mse_train": [],
        "reg_train": [],
        "mse_val": [],
        "best_mse": [],
        "best_ep": [],
        "lam_reg": [],
        "lr": []
    }

    best = -1
    previous_val = float("inf")

    warmup_epochs = int(warmup*epochs)
    
    for ep in range(1, epochs+1):
        enc.train(); dec.train()

        progress = max(ep - warmup_epochs, 0) / (epochs - warmup_epochs)
        lam_reg_current = progress * lam_reg

        mse = 0.0
        reg = 0.0
        anc = 0.0
        for x in dl_train:
            x = x.to(device)
            z = enc(x)
            xhat = dec(z)
            
            mse_loss = F.mse_loss(xhat, x)

            x_min = -torch.ones(x.shape[0], 1, 32, 32, device=device)   # corresponds to u=umin=0 everywhere
            x_min_hat = dec(enc(x_min))
            
            anc_loss = F.mse_loss(x_min_hat, x_min)
            
            # Warmup 50% of training
            if ep > warmup_epochs and reg_fn is not None:
                reg_loss = reg_fn(dec, z)
            else:
                reg_loss = torch.tensor(0.0, device=device)

            loss = mse_loss + lam_reg_current * reg_loss + w_anchor * anc_loss
            
            mse += mse_loss.item()
            reg += reg_loss.item()
            anc += anc_loss.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if exp_name == "stiefel" and ep > warmup_epochs:
                # Stiefel projection
                stiefel_project_conv_columns(dec.conv_in)            

            # End loop

            
        tr  = mse / len(dl_train)
        rg  = reg / len(dl_train)
        an  = anc / len(dl_train)
        # tr = eval_ae(dl_train, enc, dec)
        va = eval_ae(dl_val, enc, dec, device=device)
        cur_lr = opt.param_groups[0]['lr']
        print(f"AE [{exp_name}-v{V}] ep {ep:03d}, LR {cur_lr:.6e}, lam_reg {lam_reg_current:.3e} | mse {tr:.3e}, anchor {an:.3e}, reg={rg:.3e} | val  {va:.3e}")

        torch.save(enc, f"{save_dir}/enc_ep{ep}.pt")
        torch.save(dec, f"{save_dir}/dec_ep{ep}.pt")  
        
        losses["mse_train"].append(tr)
        losses["reg_train"].append(rg)
        losses["mse_val"].append(va)
        losses["lam_reg"].append(lam_reg_current)
        losses["lr"].append(cur_lr)   

        if schedule=="exponential":
            scheduler.step()
        elif schedule=="plateau":
            scheduler.step(va)

        if va < previous_val:
            torch.save(enc, f"{save_dir}/enc_best.pt")
            torch.save(dec, f"{save_dir}/dec_best.pt")  
            print(f"New best model: mse={va:.6e}, epoch = {ep}")

            best = ep
            previous_val = va
            
        losses["best_mse"].append(previous_val)
        losses["best_ep"].append(best)    

        if mse_target is not None and va <= mse_target:
            print(f"[EARLY STOPPING - ep {ep}]: val loss {va:.6e} <= {mse_target:.6e} target")
            break
        
    # End training loop
    with open(f"{save_dir}/stats.json", "w") as f:
        json.dump(losses, f)

    return enc, dec, losses





@torch.no_grad()
def stiefel_project_conv_columns(conv: torch.nn.Conv2d, eps=1e-8):
    """
    Project Conv2d weight onto the set of matrices with orthonormal columns.

    For conv weight W of shape (out_ch, in_ch, k1, k2),
    reshape to A of shape (out_ch, in_flat) with in_flat = in_ch*k1*k2.

    If out_ch >= in_flat, we project columns:
        A_proj = A (A^T A)^(-1/2)
    so that A_proj^T A_proj = I.

    If out_ch < in_flat (rare here), fall back to row-normalization.
    """
    W = conv.weight.data
    out_ch, in_ch, k1, k2 = W.shape
    A = W.reshape(out_ch, in_ch * k1 * k2)  # (out, in_flat)
    in_flat = A.shape[1]

    if out_ch < in_flat:
        # fallback (not Stiefel): normalize rows
        A = A / (A.norm(dim=1, keepdim=True).clamp_min(eps))
        conv.weight.data = A.reshape_as(W)
        return

    # Gram = A^T A : (in_flat, in_flat)
    G = A.T @ A

    # Compute G^{-1/2} via eigendecomposition (small: 9x9 in your case)
    evals, evecs = torch.linalg.eigh(G)  # evals ascending
    evals = evals.clamp_min(eps)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T  # G^{-1/2}

    A_proj = A @ inv_sqrt
    conv.weight.data = A_proj.reshape_as(W)


@torch.no_grad()
def stiefel_diagnostic_conv_columns(conv: torch.nn.Conv2d):
    """
    Returns max|A^T A - I| and Frobenius norm of (A^T A - I).
    """
    W = conv.weight.data
    out_ch, in_ch, k1, k2 = W.shape
    A = W.reshape(out_ch, in_ch * k1 * k2)
    in_flat = A.shape[1]
    G = A.T @ A
    I = torch.eye(in_flat, device=W.device, dtype=W.dtype)
    D = G - I
    return float(D.abs().max().cpu()), float(torch.linalg.norm(D).cpu())