import torchvision
import torchvision.transforms as T
import torch
import yaml
import json

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

list_loss_function = [
        "triplet",
        "MSE"
]

list_model_type = [
        "CNNencoder",
        "ResNet18",
        "PerResNet18",
        "ResNet50",
        "PerResNet50",
        "VGG11_BN",
        "PerVGG11_BN",
        "VGG13_BN",
        "PerVGG13_BN"
]

for model_type in list_model_type:
    for loss_funct in list_loss_function:

        config["training"]["loss_function"] = loss_funct

        models_path = f'./saved_models/{loss_funct}_{model_type}/'
        if not os.path.exists(models_path):
             os.makedirs(models_path)
             os.makedirs(f'{models_path}checkpoints')

        config["training"]["model_path"] = models_path

        full_dataset, train_loader, val_loader = get_train_dataset(config["train_dataset"], config["training"]["loss_function"], transform)
        _, test_loader = get_test_dataset(config["test_dataset"], transform)

        if model_type == "CNNencoder": 
            encoder = CNNencoder() 
        elif model_type == "PerResNet18": 
            encoder = PersonalizedResNet18(config["model"]) 
        elif model_type == "ResNet18":
            encoder = ResNet18(config["model"]) 
        elif model_type == "PerResNet50": 
            encoder = PersonalizedResNet50(config["model"])
        elif model_type == "ResNet50":
            encoder = ResNet50(config["model"])
        elif model_type == "VGG11_BN":
            encoder = VGG11_BN(config["model"])
        elif model_type == "PerVGG11_BN":
            encoder = PersonalizedVGG11_BN(config["model"])
        elif model_type == "VGG13_BN":
            encoder = VGG13_BN(config["model"])
        elif model_type == "PerVGG13_BN":
            encoder = PersonalizedVGG13_BN(config["model"])

        encoder.cuda()

        init_run(f"{loss_funct}_{model_type}", dict({
            "learning_rate": config["training"]["learning_rate"],
            "architecture": model_type,
            "dataset": "UTKFace",
            "dataset_size": len(full_dataset),
            "epochs": config["training"]["max_epochs"],
            "batch_size": config["train_dataset"]["batch_size"],
            "device": "cuda",
            "loss": config["training"]["loss_function"]
        }))

        print(f'Configuration: {loss_funct} - {model_type}')
        trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])
        trainer.SetupTrain()
        trainer.train(train_loader, val_loader, test_loader)
        print("")

        save_recap("runs_recap.json")



# === Best models Selection for the submit === #

def find_best_models():

    with open("runs_recap.json", "r") as read_file:
        data = json.load(read_file)

    best_models = {}   # dictionary containing model info & minimum_error for each model
    for model in data:
        min_error = min(data[model]['test_loss'])
        best_models[model] = {str(model): data[model],
                            "minimum_error": min_error}

    print(best_models)

    ##  === now to find the best 4 === ##
    global_min_loss = float('inf')
    top_four = []
    for model in best_models:
        if best_models[model]["minimum_error"] < global_min_loss:     # best model so far
            global_min_loss = best_models[model]['minimum_error']     # update the global min loss
            top_four.insert(0, model)                                 # add new best (model name) in top position


    print(best_models[top_four[0]]["minimum_error"])
    print(best_models[top_four[1]]["minimum_error"])
    print(best_models[top_four[2]]["minimum_error"])            
    print(best_models[top_four[3]]["minimum_error"])

    return top_four[:4]

#top_four = find_best_models()
#print(top_four)
