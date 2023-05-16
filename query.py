import torchvision
import torchvision.transforms as T
import torch
import yaml
import matplotlib.pyplot as plt

from dataset import TestDataset
from network import *
from trainer import *


config_path = "./config/resnet18_inet1k_init.yaml"
with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"\tConfiguration file loaded from: {config_path}")

DATA_PATH = config["test_dataset"]["data_root"]
IMG_SIZE = (config["general"]["img_size"],config["general"]["img_size"])
BATCH_SIZE = config["test_dataset"]["batch_size"]
DEVICE = torch.device("cuda")
TRANSFORM = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalization for the ResNet18
    ])

if __name__ == "__main__":

    test_dataset, test_loader = get_test_dataset(config['test_dataset'], TRANSFORM)
    

    #rsync -r -e 'ssh -p 61099' azure_dir/ disi@ml-lab-55bc589a-5fd7-4f52-b071-64c2815e9b95.westeurope.cloudapp.azure.com:/home/disi/ML_project

    encoder = PersonalizedResNet18(config["model"])
    encoder.cuda()
    encoder.load_state_dict(torch.load(f'{config["training"]["model_path"]}best.pth', map_location=DEVICE))

    trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])

    distance_list, indices_list, error = trainer.test_step(test_loader)
    print(indices_list[0])
    print(error)

    plt.figure(figsize=(20,20))
    for i,index in enumerate(indices_list[0]):
        img = test_dataset[index]
        ax=plt.subplot(3,4,i+1)
        plt.imshow(img.permute(1,2,0))
        plt.annotate(f"d: {distance_list[0][i]}", (100,0))#:.7f
        plt.axis("off")
    plt.show()