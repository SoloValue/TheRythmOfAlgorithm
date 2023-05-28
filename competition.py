# from dataset import *
# from utils import *
# from network import *
# from model_selection import *
# from query import *

import json

""" select best models to use for the submit """

###to do
# with open("runs_recap.json", "r") as read_file:
#     data = json.load(read_file)

# minimum_loss = float('inf')
# best_model = {}

# for model in data:
#     error = min(data[model]['test_loss'])
#     if error < minimum_loss:
#         minimum_loss = error
#         best_model = {model: data[model]}

# print("Minimum loss:", minimum_loss)
# print("Best model:", best_model)

##MARTA##
#sistemare per far sì che best model sia un dizionario di dizionari con i 4 modelli con i test error migliori (quindi modificare il min)

# SARA - TOP 4 MODELS
with open("runs_recap.json", "r") as read_file:
    data = json.load(read_file)

best_models = {}   # dictionary containing models

for model in data:
    min_error = min(data[model]['test_loss'])
    best_models[model] = {str(model): data[model],
                        "minimum_error": min_error}

print(best_models)

## now to find the best 4
global_min_loss = float('inf')
top_four = []
for model in best_models:
    if best_models[model]["minimum_error"] < global_min_loss:     # best model so far
        global_min_loss = best_models[model]['minimum_error']     # update the global min loss
        top_four.insert(0, model)                    # add new best (model name) in top position

#print("Minimum loss:", minimum_loss)
#print("Best model:", best_model)
print(top_four[:4])
print(best_models[top_four[0]]["minimum_error"])
print(best_models[top_four[1]]["minimum_error"])
print(best_models[top_four[2]]["minimum_error"])
print(best_models[top_four[3]]["minimum_error"])

###### SARA ######
""" feed query and gallery to model -> get results -> submit"""
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
    model_to_tun = "...TO INSERT ON COMP DAY..."             # INSERT ON COMP DAY !!!!!
    top_n = # INSERT ON COMP DAY (number of k neighbours for knn)

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
    # SARA: secondo me dobbiamo aggiustare un po' il test_step per la comp (query e gallery separate) OPPURE creare un comp_test 
    trainer = UnsupervisedTransferLearnTrainer(encoder, config["training"])  # SARA: CAMBIARE CONFIG?

    distance_list, indices_list = trainer.comp_step(query_loader, gallery_loader, top_n)
    print(indices_list[0])

    ## 'PACK UP' RESULTS AND SUBMIT THEM