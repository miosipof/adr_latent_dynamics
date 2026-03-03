# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


def norm_u(u, umin, umax):
    # u -> [-1, 1]
    return 2.0 * (u - umin) / (umax - umin) - 1.0

def inv_norm_u(u_norm, umin, umax):
    # [-1,1] -> u
    return 0.5 * (u_norm + 1.0) * (umax - umin) + umin



def get_dataset(path):
    with h5py.File(path, "r") as f:
        Uc = f["train/u_coarse"]
        Uf = f["train/u_fine"]
    
        print("train/u_coarse:", Uc.shape, Uc.dtype)  # safe: file is open
        print("train/u_fine:", Uf.shape, Uf.dtype)
    
        mu_train = f["train/mu"][...]          # small enough to load
        train_idx = f["train_idx"][...]
        val_idx = f["val_idx"][...]
        t_coarse = f["t_coarse"][...]
        t_fine = f["t_fine"][...]
        coord_grid = f["coord_grid"][...]
    
    
    Nmu, Nt_coarse, H, W = (mu_train.shape[0], len(t_coarse), coord_grid.shape[0], coord_grid.shape[1])
    print("Inferred:", (Nmu, Nt_coarse, H, W))

    return mu_train, train_idx, val_idx, t_coarse, t_fine, coord_grid, Nmu, Nt_coarse, H, W





class H5Snapshots(Dataset):
    def __init__(self, h5_path, split="train", field="u_coarse",
                 mu_indices=None, t_indices=None, return_mu=True):
        self.h5_path = h5_path
        self.split = split
        self.field = field
        self.return_mu = return_mu

        with h5py.File(self.h5_path, "r") as f:
            U = f[f"{split}/{field}"]
            self.Nmu, self.Nt, self.H, self.W = U.shape

        self.mu_indices = np.arange(self.Nmu) if mu_indices is None else np.asarray(mu_indices)
        self.t_indices  = np.arange(self.Nt)  if t_indices  is None else np.asarray(t_indices)

        self.M = len(self.mu_indices)
        self.T = len(self.t_indices)

        self._f = None  # lazy-open


    def compute_train_minmax(self, field="u_coarse", mu_ids=None, t_ids=None, split="train"):
        """
        Computes min/max over U[mu_ids, t_ids, :, :] in a streaming way.
        """
        with h5py.File(self.h5_path, "r") as f:
            U = f[f"{split}/{field}"]  # (Nmu, Nt, H, W)
            Nmu, Nt, H, W = U.shape
    
            if mu_ids is None:
                mu_ids = np.arange(Nmu, dtype=np.int64)
            else:
                mu_ids = np.asarray(mu_ids, dtype=np.int64)
    
            if t_ids is None:
                t_ids = np.arange(Nt, dtype=np.int64)
            else:
                t_ids = np.asarray(t_ids, dtype=np.int64)
    
            umin = np.inf
            umax = -np.inf
    
            for i in mu_ids:
                # read only the needed time slab for this mu
                slab = U[i, t_ids, :, :]  # (len(t_ids), H, W)
                slab_min = float(np.min(slab))
                slab_max = float(np.max(slab))
                umin = min(umin, slab_min)
                umax = max(umax, slab_max)
    
        return float(umin), float(umax)
    
    def _open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return self.M * self.T

    def __getitem__(self, idx):
        self._open()
        i_pos = idx // self.T
        k_pos = idx % self.T
        i = int(self.mu_indices[i_pos])
        k = int(self.t_indices[k_pos])

        u = self._f[f"{self.split}/{self.field}"][i, k]  # (H,W)
        u = np.array(u, dtype=np.float32, copy=False)
        u = torch.from_numpy(u)[None, ...]

        u = norm_u(u, self.umin, self.umax)

        if not self.return_mu:
            return u

        mu = self._f[f"{self.split}/mu"][i]
        mu = np.array(mu, dtype=np.float32, copy=False)
        mu = torch.from_numpy(mu)
        return u, mu


class H5SubTrajectories(Dataset):
    """
    Returns a random contiguous window of length (lmax+1) from a trajectory.
    Output shapes:
      u_win: (lmax+1, 1, ny, nx)
      t_win: (lmax+1,)
      mu:    (nmu,)  (optional)
    """
    def __init__(self, path, split="train", field="u_coarse", mu_indices=None,
                 lmax=40, return_mu=True, seed=0, fixed_k0=None):
        self.path = path
        self.split = split
        self.field = field
        self.lmax = int(lmax)
        self.return_mu = bool(return_mu)
        self.max_u = 1.0

        self.seed = int(seed)
        self.epoch = 0  # will be updated by set_epoch

        self.fixed_k0 = fixed_k0 # Fixed start for validation

        # indices of parameter instances to use
        with h5py.File(self.path, "r") as f:
            U = f[f"{split}/{field}"]
            self.Nmu, self.Nt, self.ny, self.nx = U.shape
            self.nmu = f[f"{split}/mu"].shape[1] if (return_mu and f.get(f"{split}/mu") is not None) else 0

        if mu_indices is None:
            self.mu_indices = np.arange(self.Nmu, dtype=np.int64)
        else:
            self.mu_indices = np.asarray(mu_indices, dtype=np.int64)

        if self.Nt < self.lmax + 1:
            raise ValueError(f"Nt={self.Nt} < lmax+1={self.lmax+1}")

        # Lazy-open handle (per worker/process)
        self._f = None

    def set_epoch(self, epoch: int):
        """Call once per epoch from the training loop."""
        self.epoch = int(epoch)

    def _get_file(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.mu_indices)

    def __getitem__(self, idx):
        f = self._get_file()
        mu_id = int(self.mu_indices[idx])

        U = f[f"{self.split}/{self.field}"]  # (Nmu, Nt, ny, nx)
        
        t = f["t_coarse"] if "coarse" in self.field else f["t_fine"]

        max_k0 = self.Nt - (self.lmax + 1)

        if self.fixed_k0 is not None:
            # Fixed start at k0 for validation
            k0 = int(self.fixed_k0)
        else:          
            # Deterministic, epoch-dependent window selection
            mix = (
                self.seed
                + 1009 * self.epoch
                + 1000003 * mu_id
            )
            rng = np.random.default_rng(mix)
            k0 = int(rng.integers(0, max_k0 + 1))

        u_win = U[mu_id, k0:k0 + self.lmax + 1]  # (lmax+1, ny, nx)
        u_win = norm_u(u_win, self.umin, self.umax)
        t_win = t[k0:k0 + self.lmax + 1]         # (lmax+1,)

        u_win = torch.from_numpy(np.asarray(u_win, dtype=np.float32)).unsqueeze(1)
        t_win = torch.from_numpy(np.asarray(t_win, dtype=np.float32))

        if self.return_mu:
            mu = torch.from_numpy(
                np.asarray(f[f"{self.split}/mu"][mu_id], dtype=np.float32)
            )
            return u_win, t_win, mu
        else:
            return u_win, t_win




class H5WindowedTrajectories(Dataset):
    """
    Paper-aligned NODE dataset: enumerates *all* contiguous windows (mu_id, k0)
    inside a specified time segment, instead of sampling one random window per mu.

    Each item returns a window of length (lmax+1):
      u_win: (lmax+1, 1, ny, nx)
      t_win: (lmax+1,)
      mu:    (nmu,)  (optional)

    Parameters
    ----------
    path : str
        HDF5 dataset path.
    split : str
        HDF5 group (e.g. "train").
    field : str
        "u_coarse" or "u_fine" (or any dataset shaped (Nmu, Nt, ny, nx)).
    mu_indices : array-like
        Which mu indices to use.
    lmax : int
        Maximum horizon; the dataset always returns lmax+1 snapshots.
        The training loop may slice to shorter ell for temporal regularization.
    t_start : int
        Start time index (inclusive) of the allowed segment.
    t_end : int | None
        End time index (inclusive) of the allowed segment. If None, uses Nt-1.
    return_mu : bool
        Whether to return mu with each sample.
    return_k0 : bool
        Whether to also return the start index k0 (useful for debugging).
    """
    def __init__(
        self,
        path: str,
        split: str = "train",
        field: str = "u_coarse",
        mu_indices=None,
        lmax: int = 40,
        t_start: int = 0,
        t_end=None,
        return_mu: bool = True,
        return_k0: bool = False,
    ):
        self.path = path
        self.split = split
        self.field = field
        self.lmax = int(lmax)
        self.return_mu = bool(return_mu)
        self.return_k0 = bool(return_k0)

        with h5py.File(self.path, "r") as f:
            U = f[f"{split}/{field}"]
            self.Nmu, self.Nt, self.ny, self.nx = U.shape
            if self.return_mu and f.get(f"{split}/mu") is not None:
                self.nmu = f[f"{split}/mu"].shape[1]
            else:
                self.nmu = 0

            # choose time grid key (backward compatible)
            if "coarse" in self.field:
                self._t_key = "t_coarse" if "t_coarse" in f else "time/t_coarse"
            else:
                self._t_key = "t_fine" if "t_fine" in f else "time/t_fine"

        if mu_indices is None:
            self.mu_indices = np.arange(self.Nmu, dtype=np.int64)
        else:
            self.mu_indices = np.asarray(mu_indices, dtype=np.int64)

        self.t_start = int(t_start)
        self.t_end = int(self.Nt - 1) if (t_end is None) else int(t_end)

        if not (0 <= self.t_start <= self.t_end < self.Nt):
            raise ValueError(f"Invalid time segment: t_start={self.t_start}, t_end={self.t_end}, Nt={self.Nt}")

        seg_len = self.t_end - self.t_start + 1
        if seg_len < (self.lmax + 1):
            raise ValueError(
                f"Time segment too short for lmax={self.lmax}: seg_len={seg_len} < lmax+1={self.lmax+1}."
            )

        # valid k0 ensures [k0, k0+lmax] stays inside [t_start, t_end]
        self.num_k0 = (self.t_end - self.t_start - self.lmax + 1)
        assert self.num_k0 > 0

        self._f = None  # lazy-open
        # normalization bounds must be set externally (like other datasets)
        self.umin = None
        self.umax = None

    def _get_file(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return int(len(self.mu_indices) * self.num_k0)

    def __getitem__(self, idx: int):
        f = self._get_file()

        i_pos = int(idx // self.num_k0)
        k_pos = int(idx % self.num_k0)

        mu_id = int(self.mu_indices[i_pos])
        k0 = int(self.t_start + k_pos)

        U = f[f"{self.split}/{self.field}"]  # (Nmu, Nt, ny, nx)
        t = f[self._t_key]

        u_win = U[mu_id, k0 : k0 + self.lmax + 1]  # (lmax+1, ny, nx)
        t_win = t[k0 : k0 + self.lmax + 1]         # (lmax+1,)

        if self.umin is None or self.umax is None:
            raise RuntimeError("Please set dataset.umin and dataset.umax before using H5WindowedTrajectories.")

        u_win = norm_u(u_win, self.umin, self.umax)

        u_win = torch.from_numpy(np.asarray(u_win, dtype=np.float32)).unsqueeze(1)
        t_win = torch.from_numpy(np.asarray(t_win, dtype=np.float32))

        out = [u_win, t_win]

        if self.return_mu:
            mu = torch.from_numpy(np.asarray(f[f"{self.split}/mu"][mu_id], dtype=np.float32))
            out.append(mu)

        if self.return_k0:
            out.append(torch.tensor(k0, dtype=torch.int64))

        return tuple(out)


def load_time_segment_from_h5(path: str, kind: str = "coarse", segment: str = "train"):
    """
    Convenience helper to obtain (t_start, t_end) inclusive indices for common segments.

    Requires the paper-aligned HDF5 schema:
      splits/coarse_ids_train, splits/coarse_ids_val_loop, ...
    Falls back to attrs iT1_coarse if splits group is missing.

    Returns (t_start, t_end) inclusive.
    """
    with h5py.File(path, "r") as f:
        if "splits" in f:
            if kind == "coarse":
                if segment == "train":
                    key = "splits/coarse_ids_train"
                elif segment in ("val", "val_loop"):
                    key = "splits/coarse_ids_val_loop"
                else:
                    raise ValueError(f"Unknown coarse segment: {segment}")
            else:
                if segment == "train":
                    key = "splits/fine_ids_train"
                elif segment == "interp":
                    key = "splits/fine_ids_interp"
                elif segment == "extrap":
                    key = "splits/fine_ids_extrap"
                else:
                    raise ValueError(f"Unknown fine segment: {segment}")

            ids = np.asarray(f[key][...], dtype=np.int64)
            return int(ids[0]), int(ids[-1])

        # fallback for older files (coarse only)
        if kind == "coarse":
            if "iT1_coarse" not in f.attrs:
                raise KeyError("HDF5 missing splits group and iT1_coarse attribute.")
            iT1 = int(f.attrs["iT1_coarse"])
            Nt = f["t_coarse"].shape[0]
            if segment == "train":
                return 0, iT1
            elif segment in ("val", "val_loop"):
                return iT1 + 1, Nt - 1
            else:
                raise ValueError(f"Unknown coarse segment: {segment}")

        raise KeyError("HDF5 missing required split metadata for fine segments.")






def compute_u_absmax(path, split="train", field="u_coarse", mu_indices=None, t_chunk=32):
    """
    Returns S = max |u| over given mu_indices and all timesteps/pixels, streaming over HDF5.
    """
    with h5py.File(path, "r") as f:
        U = f[f"{split}/{field}"]  # (Nmu, Nt, H, W)
        Nmu, Nt = U.shape[0], U.shape[1]

        if mu_indices is None:
            mu_indices = np.arange(Nmu, dtype=np.int64)
        else:
            mu_indices = np.asarray(mu_indices, dtype=np.int64)

        m = 0.0
        for mu_id in mu_indices:
            for k0 in range(0, Nt, t_chunk):
                k1 = min(Nt, k0 + t_chunk)
                block = U[mu_id, k0:k1, :, :]  # h5py slice
                # convert to numpy array (float32) and reduce
                block = np.asarray(block, dtype=np.float32)
                m = max(m, float(np.max(np.abs(block))))
        return m

def store_norm_stats(path, key, value):
    with h5py.File(path, "a") as f:
        f.attrs[key] = float(value)

