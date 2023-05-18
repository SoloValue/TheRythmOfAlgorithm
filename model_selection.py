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

list_loss_function = [
        "triplet",
        "MSE"
]

list_model_type = [
        "ResNet18",
        "PerResNet18",
        "ResNet101",
        "PerResNet101",
        "VGG16_BN",
        "PerVGG16_BN",
]

for loss_funct in list_loss_function:
    for model_type in list_model_type:

        config["training"]["loss_function"] = loss_funct

        models_path = f'./saved_models/{loss_funct}_{model_type}/'
        if not os.path.exists(models_path):
             os.makedirs(models_path)
             os.makedirs(f'{models_path}checkpoints')

        config["training"]["model_path"] = models_path

        full_dataset, train_loader, val_loader = get_train_dataset(config["train_dataset"], config["training"]["loss_function"], transform)
        _, test_loader = get_test_dataset(config["test_dataset"], transform)

        wandb.init(
                # Set the project where this run will be logged
                project=config["general"]["project_name"],
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=f"{loss_funct}_{model_type}", 
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

        if model_type == "CNNencoder": 
            encoder = CNNencoder(config["model"])
        elif model_type == "PerResNet18": 
            encoder = PersonalizedResNet18(config["model"])
        elif model_type == "ResNet18":
            encoder = ResNet18(config["model"])
        elif model_type == "PerResNet101": 
            encoder = PersonalizedResNet101(config["model"])
        elif model_type == "ResNet101":
            encoder = ResNet101(config["model"])
        elif model_type == "VGG16_BN":
             encoder = VGG16_BN(config["model"])
        elif model_type == "PerVGG16_BN":
             encoder = PersonalizedVGG16_BN(config["model"])

        encoder.cuda()

        trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])
        trainer.SetupTrain()
        trainer.train(train_loader, val_loader, test_loader)

        wandb.finish()