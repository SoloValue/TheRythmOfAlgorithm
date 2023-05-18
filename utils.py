import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def seed_everything(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_rand():
    return random.randint(1,100)

