import torchvision
import torchvision.transforms as T
import torch
import yaml
import wandb

from dataset import *
from utils import *
from network import *
from trainer import *

wandb.login()
#6be89a04516c30d593ca98e94d3477da24546f26

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

wandb.init(
        # Set the project where this run will be logged
        project=config["general"]["project_name"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"test_piu_bello", 
        # Track hyperparameters and run metadata
        config={
        "learning_rate": config["training"]["learning_rate"],
        "architecture": "ResNet18",
        "dataset": "UTKFace",
        "dataset_size": len(full_dataset),
        "epochs": config["training"]["max_epochs"],
        "batch_size": config["train_dataset"]["batch_size"],
        "device": "cuda",
        "loss": config["training"]["loss_function"]
        })

#rsync -r -e 'ssh -p 61099' azure_dir/ disi@ml-lab-55bc589a-5fd7-4f52-b071-64c2815e9b95.westeurope.cloudapp.azure.com:/home/disi/ML_project

encoder = PersonalizedResNet18(config["model"])
encoder.cuda()

trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])
trainer.SetupTrain()
trainer.train(train_loader, val_loader, test_loader)

