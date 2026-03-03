# src/node.py

import os, math, json, shutil
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    def __init__(self, k=8, Tmax=31.41592653589793):
        super().__init__()
        assert k % 2 == 0
        self.k = k
        self.Tmax = float(Tmax)
        half = k // 2
        # Paper Eq. (4.3): omega_j = 1 / Tmax^(j-1), j=1..half
        # => exponents 0..half-1
        exps = torch.arange(half).float()
        self.register_buffer("freqs", (1.0 / (self.Tmax ** exps)))  # (half,)

    def forward(self, x):
        if x.dim() == 0:
            x = x[None]
        if x.dim() == 1:
            xw = x[:, None] * self.freqs[None, :]
            return torch.cat([torch.sin(xw), torch.cos(xw)], dim=1)
        elif x.dim() == 2:
            outs = []
            for j in range(x.shape[1]):
                xw = x[:, j:j+1] * self.freqs[None, :]
                outs.append(torch.cat([torch.sin(xw), torch.cos(xw)], dim=1))
            return torch.cat(outs, dim=1)
        else:
            raise ValueError("x must be scalar, (B,), or (B,d).")



class AffineModulatedLatentODEFunc(nn.Module):
    """
    Implements Algorithm 2 structure: input conv -> 2 modulated convs -> output conv.
    z is treated as a 2D latent field (e.g., (B,1,4,4)).
    """
    def __init__(self, nmu, k=8, Tmax=31.41592653589793, de=64, nc=32, z_channels=1):
        super().__init__()
        self.nmu = nmu
        self.time_enc = SinusoidalEncoding(k=k, Tmax=Tmax)
        self.mu_enc   = SinusoidalEncoding(k=k, Tmax=Tmax)

        in_dim = k + k * nmu
        self.embed = nn.Sequential(
            nn.Linear(in_dim, de),
            nn.ELU(),
            nn.Linear(de, de),
            nn.ELU(),
        )

        # produces gamma1,gamma2,beta1,beta2 each in R^nc  => total 4*nc
        self.to_gb = nn.Linear(de, 4 * nc)

        # dynamics convs
        self.conv_in  = nn.Conv2d(z_channels, nc, kernel_size=3, padding=1)
        self.conv1    = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(nc, nc, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(nc, z_channels, kernel_size=3, padding=1)

    def forward(self, t, z, mu):
        """
        t:  (B,) or scalar
        z:  (B,1,4,4)
        mu: (B,nmu)
        returns z_dot with same shape as z
        """
        B = z.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=z.device, dtype=z.dtype)
        if t.dim() == 0:
            t = t.expand(B)

        t_emb  = self.time_enc(t)      # (B,k)
        mu_emb = self.mu_enc(mu)       # (B,k*nmu)
        xi = self.embed(torch.cat([t_emb, mu_emb], dim=1))  # (B,de)

        gb = self.to_gb(xi)  # (B,4*nc)
        gb = gb.view(B, 4, -1)  # (B,4,nc)
        gamma1, gamma2, beta1, beta2 = gb[:,0], gb[:,1], gb[:,2], gb[:,3]  # each (B,nc)

        # reshape for channel-wise affine
        g1 = gamma1[:, :, None, None]
        g2 = gamma2[:, :, None, None]
        b1 = beta1[:, :, None, None]
        b2 = beta2[:, :, None, None]

        h = torch.tanh(self.conv_in(z))
        h = torch.tanh(g1 * self.conv1(h) + b1)
        h = torch.tanh(g2 * self.conv2(h) + b2)
        z_dot = self.conv_out(h)
        return z_dot


def rk2_ralston_step(f, t, z, dt, mu):
    # dt is broadcastable to z
    k1 = f(t, z, mu)
    k2 = f(t + (2.0/3.0)*dt.squeeze(), z + (2.0/3.0)*dt*k1, mu)
    return z + dt * (0.25*k1 + 0.75*k2)

def rk4_classic_step(f, t, z, dt, mu):
    k1 = f(t, z, mu)
    k2 = f(t + 0.5*dt.squeeze(), z + 0.5*dt*k1, mu)
    k3 = f(t + 0.5*dt.squeeze(), z + 0.5*dt*k2, mu)
    k4 = f(t + 1.0*dt.squeeze(), z + 1.0*dt*k3, mu)
    return z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

def integrate_latent(f, z0, t_grid, mu, method="rk2"):
    """
    Differentiable integrator (NO @torch.no_grad).
    z0: (B,1,4,4)
    t_grid: (B,L)
    mu: (B,nmu)
    returns: (B,L,1,4,4)
    """
    B, L = t_grid.shape
    z = z0
    zs = [z0]

    for i in range(L - 1):
        t = t_grid[:, i]                                # (B,)
        dt = (t_grid[:, i+1] - t_grid[:, i]).view(B,1,1,1)

        if method == "rk2":
            z = rk2_ralston_step(f, t, z, dt, mu)
        elif method == "rk4":
            z = rk4_classic_step(f, t, z, dt, mu)
        else:
            raise ValueError("method must be 'rk2' or 'rk4'")

        zs.append(z)

    return torch.stack(zs, dim=1)


@torch.no_grad()
def integrate_latent_eval(f, z0, t_grid, mu, method="rk2"):
    # Same code but wrapped for eval convenience
    return integrate_latent(f, z0, t_grid, mu, method=method)
    
# -------------------------
# Helpers: LR warmup, curriculum, deterministic ell, loader factory
# -------------------------

def _as_epochs(x, total_epochs: int) -> int:
    """
    Accepts:
      - int >= 0  -> epochs
      - float in (0,1] -> fraction of total_epochs
      - None -> 0
    """
    if x is None:
        return 0
    if isinstance(x, float):
        if x <= 0:
            return 0
        return int(max(1, round(x * total_epochs)))
    return int(max(0, x))


def lr_factor_with_warmup_and_exp(ep: int, warmup_epochs: int, gamma: float):
    """
    ep is 1-indexed.
    During warmup: lr scales linearly from 0 -> 1.
    After warmup: lr follows exponential decay gamma^(ep-warmup).
    """
    if warmup_epochs > 0 and ep <= warmup_epochs:
        return ep / float(warmup_epochs)
    return gamma ** max(ep - warmup_epochs, 0)


def curriculum_ell_cap(ep: int, curriculum_epochs: int, lmax: int, ell_min: int = 2) -> int:
    """
    Linearly ramps the maximum allowed ell from ell_min to lmax over curriculum_epochs.
    ep is 1-indexed.
    """
    if curriculum_epochs <= 0:
        return lmax
    frac = min(1.0, max(0.0, (ep - 1) / float(max(1, curriculum_epochs - 1))))
    cap = int(round(ell_min + frac * (lmax - ell_min)))
    return int(np.clip(cap, ell_min, lmax))


def deterministic_randint(low: int, high: int, seed: int, epoch: int, batch_idx: int, device: str):
    """
    Deterministic "torch.randint" replacement producing a single int in [low, high).
    Uses an independent Generator whose seed depends on (seed, epoch, batch_idx).
    """
    g = torch.Generator(device=device if device.startswith("cuda") else "cpu")
    # Large, relatively prime-ish mix constants:
    mixed = int(seed) + 1009 * int(epoch) + 9176 * int(batch_idx)
    g.manual_seed(mixed)
    # generator must be on CPU for CPU tensors; for CUDA device, torch.randint supports CPU generator too.
    return int(torch.randint(low=low, high=high, size=(1,), generator=g, device=device).item())


def make_seeded_loader(dataset, batch_size, shuffle, seed, **dl_kwargs):
    """
    IMPORTANT for paired comparisons:
    Make a fresh DataLoader per method (or per epoch) using the SAME seed,
    so shuffle order is identical across methods.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        **dl_kwargs
    )


# -------------------------
# Updated training function
# -------------------------

def train_latent_dynamics(
    enc, dec, ode_func,
    dl_train, dl_val,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=20,

    # (1) LR warmup
    warmup=0.1,               # int epochs OR fraction in (0,1]; set 0 to disable
    lr=1e-3,
    weight_decay=1e-6,
    schedule="exponential",   # "exponential" or "plateau"
    exp_gamma=0.95,          # for exponential schedule (only used if schedule=="exponential")

    lmax=40,
    rk_method="rk4",

    train_ae=False,
    reg_fn=None,
    lam_reg=0.0,
    lam_recon=1.0,
    use_amp=False,
    grad_clip=1.0,            # (2) gradient clipping (max norm); set <=0 to disable

    # (4) curriculum on horizon
    curriculum=1.0,           # int epochs OR fraction in (0,1]; set 0 to disable

    # (5) deterministic ell sampling
    deterministic_ell=True,
    seed=0,                   # controls deterministic ell sampling

    # optional: load initialization for ode_func (shared init)
    init_ode_state_dict=None,

    stiefel_project_fn=None,
    exp_name="vanilla",
    V=1,
    run=1,
    save_root="output/NODE",
    save_dir_override=None,   # set if you want a custom save dir
    w_anchor=0.1
):
    save_dir = save_dir_override or f"{save_root}/{exp_name}_v{V}/run_{run}"

    try:
        shutil.rmtree(save_dir)
    except Exception:
        pass
    os.makedirs(save_dir, exist_ok=True)

    losses = {"mse_train": [], "rec_train": [], "reg_train": [], "mse_val": [], "best_mse": [], "best_ep": [], "lr": []}
    best_ep = -1
    best_val = float("inf")

    enc, dec, ode_func = enc.to(device), dec.to(device), ode_func.to(device)

    # shared init
    if init_ode_state_dict is not None:
        ode_func.load_state_dict(init_ode_state_dict, strict=True)

    # freeze AE if desired
    if not train_ae:
        for p in enc.parameters(): p.requires_grad = False
        for p in dec.parameters(): p.requires_grad = False

    trainable_params = list(ode_func.parameters())        
    
    if train_ae:
        trainable_params += list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam([
            {"params": ode_func.parameters(), "lr": lr},
            {"params": list(enc.parameters()) + list(dec.parameters()), "lr": lr * 0.1},
        ], weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        

    # plateau scheduler (optional)
    plateau_sched = None
    if schedule == "plateau":
        plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10
        )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))
    autocast = torch.cuda.amp.autocast if (use_amp and device.startswith("cuda")) else nullcontext

    warmup_epochs = _as_epochs(warmup, epochs)
    
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    def set_lr_for_epoch(ep: int):
        if schedule == "exponential":
            factor = lr_factor_with_warmup_and_exp(ep, warmup_epochs, exp_gamma)
            for pg, base in zip(opt.param_groups, base_lrs):
                pg["lr"] = base * factor
        else:
            # plateau: do warmup only, then plateau scheduler controls it
            if warmup_epochs > 0 and ep <= warmup_epochs:
                factor = ep / float(warmup_epochs)
                for pg, base in zip(opt.param_groups, base_lrs):
                    pg["lr"] = base * factor
    
    def run_epoch(dl, ep: int, train: bool):
        ode_func.train(train)
        if train_ae:
            enc.train(train); dec.train(train)
        else:
            enc.eval(); dec.eval()
    
        total = 0.0
        recon = 0.0
        anc = 0.0
        reg = 0.0
        n = 0

    
        for batch_idx, batch in enumerate(dl):
            u_win, t_win, mu = batch
            u_win = u_win.to(device)
            t_win = t_win.to(device)
            mu    = mu.to(device)

            # Temporal regularization: sample ell ~ U[2, lmax] each iteration`
            if train:
                if deterministic_ell:
                    ell = deterministic_randint(2, lmax+1, seed=seed, epoch=ep, batch_idx=batch_idx, device=device)
                else:
                    ell = int(torch.randint(2, lmax+1, (1,), device=device).item())
            else:
                ell = lmax
          
    
            u = u_win[:, :ell+1]   # (B, Hmax+1, 1, 32, 32)
            t = t_win[:, :ell+1]   # (B, Hmax+1)
            u0 = u[:, 0]

            # print("u0:", u0.min(), u0.max(), u0.mean())
            # with torch.no_grad():
            #     urec = dec(enc(u0))
            # print("||u0||", torch.linalg.norm(u0).item(), "||urec-u0||", torch.linalg.norm(urec-u0).item())
            # import sys
            # sys.exit()
    
            if train:
                opt.zero_grad(set_to_none=True)
    
            ctx = nullcontext() if train else torch.no_grad()
            with ctx:
                with autocast():
                    z0 = enc(u0)
                    z_traj = integrate_latent(ode_func, z0, t, mu, method=rk_method) if train else \
                             integrate_latent_eval(ode_func, z0, t, mu, method=rk_method)
    
                    B_, L_ = z_traj.shape[:2]
                    z_flat = z_traj.reshape(B_*L_, *z_traj.shape[2:])
                    u_hat  = dec(z_flat).reshape(B_, L_, 1, 32, 32)

                    x_min = -torch.ones(u0.shape[0], 1, 32, 32, device=device)   # corresponds to u=umin=0 everywhere
                    x_min_hat = dec(enc(x_min))
                    anc_loss = F.mse_loss(x_min_hat, x_min)
                    
                    if ep > warmup_epochs and train and reg_fn is not None and train_ae:
                        z_last = z_traj[:, -1, ...] #.detach()
                        reg_loss = reg_fn(dec, z_last)
                    else:
                        reg_loss = torch.tensor(0.0, device=device)

                    
                    u_flat = u.reshape(B_*L_, 1, 32, 32)
                    u_rec  = dec(enc(u_flat)).reshape(B_, L_, 1, 32, 32)
                    
                    loss_roll = (u_hat - u).pow(2).mean(dim=(2,3,4)).mean()
                    loss_rec  = (u_rec - u).pow(2).mean(dim=(2,3,4)).mean()            
                    
                    loss = loss_roll + lam_recon*loss_rec + lam_reg*reg_loss + w_anchor * anc_loss

            
            if train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    opt.step()
    
                if stiefel_project_fn is not None and train_ae:
                    stiefel_project_fn(dec.conv_in)
    
            total += float(loss.detach().cpu()) * u.shape[0]
            recon += float(loss_rec.detach().cpu()) * u.shape[0]
            anc   += float(anc_loss.detach().cpu()) * u.shape[0]
            reg   += float(reg_loss.detach().cpu()) * u.shape[0]
            n += u.shape[0]
    
        return total / max(n, 1), recon / max(n, 1), anc / max(n, 1), reg / max(n, 1)


    # training loop
    for ep in range(1, epochs + 1):
        # dataset epoch controls deterministic windows (your H5SubTrajectories)
        if hasattr(dl_train, "dataset") and hasattr(dl_train.dataset, "set_epoch"):
            dl_train.dataset.set_epoch(ep)

        set_lr_for_epoch(ep)

        tr, rec, an, reg = run_epoch(dl_train, ep=ep, train=True)
        va, _, _, _    = run_epoch(dl_val,   ep=ep, train=False)

        cur_lr = opt.param_groups[0]["lr"]
        print(f"{exp_name} | AE v{V} | NODE-{run} [{ep:03d}] LR={cur_lr:.6e} recon={rec:.6e} anchor={an:.6e} reg={reg:.6e} train={tr:.6e} | val={va:.6e}")

        # plateau scheduler uses VAL
        if schedule == "plateau" and plateau_sched is not None and ep > warmup_epochs:
            plateau_sched.step(va)

        torch.save(ode_func, f"{save_dir}/ode_{ep}.pt")

        if train_ae:
            torch.save(enc, f"{save_dir}/enc_ft_{ep}.pt")
            torch.save(dec, f"{save_dir}/dec_ft_{ep}.pt")

        losses["mse_train"].append(tr)
        losses["rec_train"].append(rec)
        losses["reg_train"].append(reg)
        losses["mse_val"].append(va)
        losses["lr"].append(cur_lr)

        if va < best_val:
            torch.save(ode_func, f"{save_dir}/ode_best.pt")
            print(f"New best model: mse={va:.6e}, epoch = {ep}")
            best_val = va
            best_ep = ep
            if train_ae:
                torch.save(enc, f"{save_dir}/enc_ft_best.pt")
                torch.save(dec, f"{save_dir}/dec_ft_best.pt")            

        losses["best_mse"].append(best_val)
        losses["best_ep"].append(best_ep)

    with open(f"{save_dir}/stats.json", "w") as f:
        json.dump(losses, f)

    return ode_func, losses


# -------------------------
# Shared init: vanilla pretrain for 10 epochs
# -------------------------

def get_shared_init_from_vanilla(
    enc_van, dec_van, ode_func_van,
    dl_train, dl_val,
    init_epochs=10,
    **train_kwargs
):
    """
    Train vanilla for init_epochs, return ode_func.state_dict() as shared init.
    """
    ode_trained, _ = train_latent_dynamics(
        enc_van, dec_van, ode_func_van,
        dl_train=dl_train, dl_val=dl_val,
        epochs=init_epochs,
        exp_name="vanilla_init",
        save_dir_override=f"{train_kwargs.get('save_root','output/NODE')}/_shared_init_v{train_kwargs['V']}/vanilla_{init_epochs}ep",
        **train_kwargs
    )
    return {k: v.detach().clone() for k, v in ode_trained.state_dict().items()}

