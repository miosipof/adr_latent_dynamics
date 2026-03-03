import torch
from torch.func import jvp, vjp
import torch.nn.functional as F


def iso_penalty_exact(dec, z, detach_z=True):
    """
    Exact isometry penalty for decoder:
        L_iso = mean_B || J(z)^T J(z) - I ||_F^2
    where J(z) is Jacobian of vec(dec(z)) wrt vec(z).

    z: (B, 1, 4, 4)
    returns: (L_iso_scalar, Gram_B_16x16)
    """
    if detach_z:
        z = z.detach()  # decoder-only regularization / metric

    B = z.shape[0]
    n = z[0].numel()  # 16
    assert z.shape[1:] == (1, 4, 4), f"Expected z shape (B,1,4,4), got {z.shape}"

    # f: (B,1,4,4) -> (B,1024)
    f = lambda zz: dec(zz).flatten(1)

    eye = torch.eye(n, device=z.device, dtype=z.dtype)  # (16,16)
    Jcols = []

    # Compute J e_i for i=1..n via JVP (exact, since e_i are basis vectors)
    for i in range(n):
        v_i = eye[i].view(1, 1, 4, 4).expand(B, 1, 4, 4)  # (B,1,4,4)
        _, Jv = jvp(f, (z,), (v_i,))                      # (B,1024)
        Jcols.append(Jv)

    # Stack columns -> J: (B,1024,16)
    J = torch.stack(Jcols, dim=-1)

    # Gram = J^T J: (B,16,16)
    Gram = J.transpose(1, 2) @ J

    I = torch.eye(n, device=z.device, dtype=z.dtype).expand(B, n, n)
    L_iso = ((Gram - I) ** 2).sum(dim=(1, 2)).mean()
    
    return L_iso #, Gram





def iso_penalty_stochastic(dec, z, R=4, detach_z=True, probe="gaussian"):
    """
    Hutchinson estimator for ||J^T J - I||_F^2:
      E || (J^T J - I) v ||^2  =  ||J^T J - I||_F^2   for v~N(0,I) or Rademacher.

    Returns scalar loss (averaged over batch and probes).
    """
    def _dec_flat(dec, z):
        # z: (B,1,4,4) -> y: (B,1024)
        return dec(z).flatten(1)
    
    def _JTJ_times_v(dec, z, v):
        """
        Compute (J^T J) v using 1 JVP + 1 VJP.
    
        z: (B,1,4,4)
        v: (B,1,4,4)
        returns: (B,1,4,4)
        """
        f = lambda zz: _dec_flat(dec, zz)   # (B,1024)
    
        # VJP closure at z
        _, vjp_fn = vjp(f, z)
    
        # JVP gives J v in output space
        _, Jv = jvp(f, (z,), (v,))         # (B,1024)
    
        # VJP of Jv gives J^T (J v)
        (JTJv,) = vjp_fn(Jv)               # (B,1,4,4)
        return JTJv
        
    # Detach z to regularize decoder only (recommended)
    z0 = z.detach() if detach_z else z
    # Make sure vjp can track ops w.r.t. input (safe even if we don't use dz grads)
    z0 = z0.requires_grad_(True)

    loss = 0.0
    for _ in range(R):
        if probe == "rademacher":
            v = (torch.randint(0, 2, z0.shape, device=z0.device, dtype=torch.int8) * 2 - 1).to(z0.dtype)
        elif probe == "gaussian":
            v = torch.randn_like(z0)
        else:
            raise ValueError("probe must be 'rademacher' or 'gaussian'")

        JTJv = _JTJ_times_v(dec, z0, v)
        Av = JTJv - v
        # squared norm per sample, then average
        loss = loss + Av.flatten(1).pow(2).sum(dim=1).mean()

    return loss / R







def operator_norm_loss(dec, z, R=4, eps_fd=1e-3, alpha=1.0, detach_z=True,
                         probe="gaussian", reduction="mean"):
    """
    L_frobenius = E_v ( || J_z[v] ||_2 - alpha )^2
    where v is random and unit-normalized per sample.

    We approximate J_z[v] via finite differences:
      J_z[v] ≈ (D(z + eps*v) - D(z)) / eps

    Args:
      dec: decoder module
      z:   (B,1,4,4) latent tensor
      R:   number of probes per batch
      eps_fd: finite difference epsilon
      alpha: target magnitude (scalar)
      detach_z: if True, regularize decoder only (recommended)
      probe: 'gaussian' or 'rademacher'
      reduction: 'mean' or 'none'

    Returns:
      loss (scalar if reduction='mean', else (B,))
      stats dict with avg/std/min/max of ||Jv|| per probe averaged over probes
    """

    @torch.no_grad()
    def _unit_random_like(z, probe="gaussian", eps=1e-12):
        """
        Sample random v with same shape as z and normalize per-sample to unit norm.
        """
        if probe == "gaussian":
            v = torch.randn_like(z)
        elif probe == "rademacher":
            v = (torch.randint(0, 2, z.shape, device=z.device, dtype=torch.int8) * 2 - 1).to(z.dtype)
        else:
            raise ValueError("probe must be 'gaussian' or 'rademacher'")
    
        nrm = v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
        return v / nrm

    
    if detach_z:
        z0 = z.detach()
    else:
        z0 = z

    # We need gradients through decoder params, but not necessarily through z
    # Finite differences uses forward passes only, so no special requires_grad on z needed.
    norms_all = []

    y0 = dec(z0)                        # (B,1,32,32)    

    loss_accum = 0.0
    for _ in range(R):
        v = _unit_random_like(z0, probe=probe)

        # Forward passes
        y1 = dec(z0 + eps_fd * v)

        Jv_approx = (y1 - y0) / eps_fd      # (B,1,32,32)

        # ambient L2 norm per sample
        nrm = Jv_approx.flatten(1).norm(dim=1)  # (B,)
        norms_all.append(nrm.detach())

        loss_accum = loss_accum + (nrm - float(alpha)).pow(2)

    loss_vec = loss_accum / R  # (B,)

    norms = torch.stack(norms_all, dim=0).mean(dim=0)  # average over probes -> (B,)
    stats = {
        "Jv_norm_mean": float(norms.mean().cpu()),
        "Jv_norm_std":  float(norms.std(unbiased=False).cpu()),
        "Jv_norm_min":  float(norms.min().cpu()),
        "Jv_norm_max":  float(norms.max().cpu()),
    }

    if reduction == "mean":
        return loss_vec.mean() #, stats
    elif reduction == "none":
        return loss_vec #, stats
    else:
        raise ValueError("reduction must be 'mean' or 'none'")



def curvature_penalty(dec, z, R=4, eps_shift=1e-2, detach_z=True,
                      probe="gaussian", divide_by_eps=True, reduction="mean"):
    """
    L_curvature = E_v || J_{z+eps*v}[v] - J_z[v] ||^2
    where v is random unit vector per-sample.

    - Finite difference is in the *basepoint* z -> z+eps*v.
    - J_z[v] itself is computed exactly via JVP.

    Args:
      dec: decoder module
      z: (B,1,4,4)
      R: number of random probes
      eps_shift: epsilon for shifting basepoint (controls scale of curvature probed)
      detach_z: if True, regularize decoder only (recommended)
      divide_by_eps: if True, use (J_{z+epsv}[v]-J_z[v]) / eps  (≈ Hessian[v,v])
      reduction: 'mean' or 'none' (per-sample)

    Returns:
      loss (scalar or (B,))
      stats dict
    """

    @torch.no_grad()
    def _unit_random_like(z, probe="gaussian", eps=1e-12):
        if probe == "gaussian":
            v = torch.randn_like(z)
        elif probe == "rademacher":
            v = (torch.randint(0, 2, z.shape, device=z.device, dtype=torch.int8) * 2 - 1).to(z.dtype)
        else:
            raise ValueError("probe must be 'gaussian' or 'rademacher'")
        nrm = v.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
        return v / nrm
        
    z0 = z.detach() if detach_z else z
    B = z0.shape[0]

    # f: (B,1,4,4) -> (B,1024)
    f = lambda zz: dec(zz).flatten(1)

    loss_accum = 0.0
    diffs_all = []

    for _ in range(R):
        v = _unit_random_like(z0, probe=probe)                  # (B,1,4,4)

        # J_z[v]
        _, Jv0 = jvp(f, (z0,), (v,))                            # (B,1024)

        # Shift basepoint: z1 = z + eps*v
        z1 = z0 + float(eps_shift) * v

        # J_{z+eps*v}[v]
        _, Jv1 = jvp(f, (z1,), (v,))                            # (B,1024)

        dJv = (Jv1 - Jv0)
        if divide_by_eps:
            dJv = dJv / float(eps_shift)

        # norm in ambient space per sample
        nrm = dJv.norm(dim=1)                                   # (B,)
        diffs_all.append(nrm.detach())

        loss_accum = loss_accum + nrm.pow(2)                    # squared norm

    loss_vec = loss_accum / R                                   # (B,)

    diffs = torch.stack(diffs_all, dim=0).mean(dim=0)            # (B,)
    stats = {
        "dJv_norm_mean": float(diffs.mean().cpu()),
        "dJv_norm_std":  float(diffs.std(unbiased=False).cpu()),
        "dJv_norm_min":  float(diffs.min().cpu()),
        "dJv_norm_max":  float(diffs.max().cpu()),
        "eps_shift": float(eps_shift),
        "divide_by_eps": bool(divide_by_eps),
    }

    if reduction == "mean":
        return loss_vec.mean() #, stats
    elif reduction == "none":
        return loss_vec #, stats
    else:
        raise ValueError("reduction must be 'mean' or 'none'")