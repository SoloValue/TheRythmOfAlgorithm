import yaml
import torch
import torchvision
import torchvision.transforms as T
import requests

from dataset import TestDataset, TestLoader, get_comp_dataset
from utils import *
from network import *
from trainer import UnsupervisedTransferLearnTrainer


# === This file is for: choose model -> feed query and gallery to model -> get results -> submit === #

##PARAMETERS
MODEL_PATH = './saved_models/triplet_PerResNet18/best.pth'
model_to_run = "PerResNet18"   # Insert on competition day
top_n = 10                     # Insert on competition day (number of k neighbours for knn)

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

## === Model Selection == ##

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
elif model_type == "VGG13_BN":
    encoder = VGG13_BN(config["model"])
elif model_type == "PerVGG13_BN":
    encoder = PersonalizedVGG13_BN(config["model"])

## PREPARE
query_dataset, query_loader, gallery_dataset, gallery_loader = get_comp_dataset(config['competition_code'], TRANSFORM)

encoder.cuda()
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
print("\tModel loaded")

## FEED QUERY AND GALLERY TO MODEL
trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])  

results, distances = trainer.comp_step(query_loader, gallery_loader, top_n)

## 'PACK UP' RESULTS AND SUBMIT THEM
mydata = dict()
mydata["groupname"] = f"The Rythm of Algorithm"   

res = dict()
res = results

mydata["images"] = res
#print(mydata)

## === Final Results Submission === #
def submit(results, url="https://competition-production.up.railway.app/results/"): 
    res = json.dumps(results) 
    response = requests.post(url, res) 
    try: 
        result = json.loads(response.text) 
        print(f"accuracy is {result['results']}") 
        return result 
    except json.JSONDecodeError: 
        print(f"ERROR: {response.text}") 
        return None

result = submit(mydata)

## === save results in a .json file named "comp_results.json" === ##
if result: 
        ress = dict()
        ress[model_to_run] = dict({"accuracy": result['results'], "results": mydata["images"], "distances": distances})

        with open("comp_results.json", 'w') as fp:
                json.dump(ress, fp, indent=4)
        print("Results saved in .json!")
