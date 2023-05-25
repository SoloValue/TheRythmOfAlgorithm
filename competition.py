from dataset import *
from utils import *
from network import *
from model_selection import *
from query import *

""" query and gallery retrieval """
data_path = config['competition_code']['data_root']

""" here we want to select the model to submit """


...

###to do
with open("runs_recap_try.json", "r") as read_file:
    data = json.load(read_file)

minimum_loss = float('inf')
best_models = {}

for model in data:
    error = min(data[model]['test_loss'])
    if error < minimum_loss:
        minimum_loss = error
        best_models = {model: data[model]}

print("Minimum loss:", minimum_loss)
print("Best model:", best_models)