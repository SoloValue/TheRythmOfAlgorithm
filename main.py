import torchvision
import torchvision.transforms as T
import torch
import yaml

from dataset import *
from utils import *
from network import *
from trainer import *

seed_everything()

config_path = "./config/resnet18_inet1k_init.yaml"
with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {config_path}")

transform = T.Compose([
    T.Resize((config["general"]["img_size"],config["general"]["img_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalization for the ResNet18
    ])

full_dataset, train_loader, val_loader = get_train_dataset(config["train_dataset"], config["training"]["loss_function"], transform)
_, test_loader = get_test_dataset(config["test_dataset"], transform)

encoder = PersonalizedResNet18(config["model"])
encoder.cuda()

trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])
trainer.SetupTrain()
trainer.train(train_loader, val_loader, test_loader)