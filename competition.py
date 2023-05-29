from dataset import *
from utils import *
from network import *
from model_selection import *
from query import *
from trainer import *

###### SARA ######
""" This file is for: choose model -> feed query and gallery to model -> get results -> submit """

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

if __name__ == "__main__":
    ## SELECT MODEL
    model_to_run = "ResNet18"             # INSERT ON COMP DAY !!!!!
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

    #rsync -r -e 'ssh -p 61099' azure_dir/ disi@ml-lab-55bc589a-5fd7-4f52-b071-64c2815e9b95.westeurope.cloudapp.azure.com:/home/disi/ML_project

    encoder.cuda()
    encoder.load_state_dict(torch.load(f'{config["training"]["model_path"]}best.pth', map_location=DEVICE))

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