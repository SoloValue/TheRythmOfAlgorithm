import yaml
import torch
import torchvision
import torchvision.transforms as T

from dataset import TestDataset, TestLoader, get_comp_dataset
from utils import *
from network import *
from trainer import UnsupervisedTransferLearnTrainer

###### SARA ######
""" This file is for: choose model -> feed query and gallery to model -> get results -> submit """

config_path = "./config/resnet18_inet1k_init.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
    print(f"\tConfiguration file for competition loaded from: {config_path}")

# query and gallery retrieval
DATA_PATH = config['competition_code']['data_root']
QUERY_PATH = config['competition_code']['query_root']
GALLERY_PATH = config['competition_code']['gallery_root']
# set up
IMG_SIZE = (config["general"]["img_size"],config["general"]["img_size"])
BATCH_SIZE = config["competition_code"]["batch_size"]
DEVICE = torch.device("cuda")
TRANSFORM = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalization for the ResNet18
    ])

""" give as input to model """ 

## SELECT MODEL
model_to_run = "PerResNet18"             # INSERT ON COMP DAY !!!!!
top_n = 10    # INSERT ON COMP DAY (number of k neighbours for knn)

if model_to_run == "CNNencoder": 
    encoder = CNNencoder() 
elif model_to_run == "PerResNet18": 
    encoder = PersonalizedResNet18(config["model"]) 
elif model_to_run == "ResNet18":
    encoder = ResNet18(config["model"]) 
elif model_to_run == "PerResNet50": 
    encoder = PersonalizedResNet50(config["model"])
elif model_to_run == "ResNet50":
    encoder = ResNet50(config["model"])
elif model_to_run == "VGG11_BN":
    encoder = VGG11_BN(config["model"])
elif model_to_run == "PerVGG11_BN":
    encoder = PersonalizedVGG11_BN(config["model"])

## PREPARE
query_dataset, query_loader, gallery_dataset, gallery_loader = get_comp_dataset(config['competition_code'], TRANSFORM)

encoder.cuda()
encoder.load_state_dict(torch.load(f'{config["training"]["model_path"]}best.pth', map_location=DEVICE))
print("\tModel loaded")
## FEED QUERY AND GALLERY TO MODEL
# SARA: ho creato comp_step nel trainer.py, uguale al test_step ma lavora con query e gallery in due directory separate
trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])  # SARA: CAMBIARE CONFIG?

results = trainer.comp_step(query_loader, gallery_loader, top_n)

## 'PACK UP' RESULTS AND SUBMIT THEM
final_results = dict()
final_results["groupname"] = "The Rythm of Algorithm - ADD MODEL NAME"   # ADD MODEL NAME SO WE KNOW WHICH ONE IT IS IN THE CLASSIFICA
# TO FINISH ON COMPETITION DAY DEPENDING ON WHAT FORM THEY WANT FOR THE RANKING
final_results["ranking"] = results
print(final_results)

## SUBMIT final_results