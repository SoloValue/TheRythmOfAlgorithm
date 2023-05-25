# from dataset import *
# from utils import *
# from network import *
# from model_selection import *
# from query import *

import json
""" query and gallery retrieval """
#data_path = config['competition_code']['data_root']

""" here we want to select the model to submit """


...

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