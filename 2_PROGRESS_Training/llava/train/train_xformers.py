# Make it more memory efficient by monkey patching the LLaMA model with xformers attention.

# Need to call this before importing transformers.
import torch
from llava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

seed = 42
# PyTorch random number generator
torch.manual_seed(seed)
# CUDA random number generator
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU
# Python random number generator
import random
random.seed(seed)
# NumPy random number generator
import numpy as np
np.random.seed(seed)
# Additional settings for reproducibility
torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
torch.backends.cudnn.benchmark = False     # ensures deterministic behavior
# Optional: Set environment variable for any libraries that check it
import os
os.environ['PYTHONHASHSEED'] = str(seed)
from train import train

if __name__ == "__main__":
    train()
