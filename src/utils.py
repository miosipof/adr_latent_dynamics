import os
import random
import numpy as np
import torch

def seed_everything(seed: int, deterministic: bool = True):
    # Python / NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make CUDA/cuDNN (mostly) deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms where possible (may throw if op is nondeterministic)
        torch.use_deterministic_algorithms(True)

        # Needed for determinism in some CUDA matmul paths (esp. Ampere+)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
