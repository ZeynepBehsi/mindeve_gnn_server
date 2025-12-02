"""
Random seed fixing for reproducibility
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fix all random seeds
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Deterministic behavior (약간 느려질 수 있음)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed set to {seed}")