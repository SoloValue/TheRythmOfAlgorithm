import numpy as np
import random
import torch
import json


def seed_everything(seed=0):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_rand():
    return random.randint(1,100)

recap = dict()
global current_run
current_run = None

def init_run(name, config):
    global current_run
    current_run = name
    recap[current_run] = dict({"config":config, "train_loss":[], "val_loss":[], "test_loss":[]})

def log_run(train_loss, val_loss, test_loss):
    recap[current_run]["train_loss"].append(train_loss)
    recap[current_run]["val_loss"].append(val_loss)
    recap[current_run]["test_loss"].append(test_loss)

def print_recap():
    print(recap)

def save_recap(data_path):
    with open(data_path, 'w') as fp:
        json.dump(recap, fp)