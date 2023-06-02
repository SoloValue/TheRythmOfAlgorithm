import matplotlib.pyplot as plt
from matplotlib import ticker
import json
import torchvision
import torchvision.transforms as T
import torch
import yaml
import PIL 
import os

from dataset import TestDataset, TestLoader, get_comp_dataset

config_path = "./config/resnet18_inet1k_init.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# load results as a dictionary
ress = json.load(open('comp_results.json', 'r'))
# load image datasets & paths 
query_dataset, query_loader, gallery_dataset, gallery_loader = get_comp_dataset(config['competition_code'], transform=None)
query_path = config["competition_code"]["query_root"]
query_names = list(ress['PerResNet18']['results'].keys())
gallery_path = config["competition_code"]["gallery_root"]

# plot results for first 3 queries in comp_results.json
num_queries = 3
num_similar_images = 10
fig, axes = plt.subplots(num_queries, num_similar_images + 1, figsize=(20, 20))

for q,query_name in enumerate(query_names[0:3]):      
    img_path = os.path.join(query_path + query_name)
    img = PIL.Image.open(img_path).convert('RGB')

    axes[q, 0].imshow(img)
    axes[q, 0].text(0.5, 1.05, f'Query {q}', ha='center', va='bottom', transform=axes[q, 0].transAxes, fontsize= 8,fontstretch='ultra-condensed')
    #axes[q, 0].annotate(f'Query {q}', (100,0))
    axes[q, 0].axis('off')

    # Alternative..
    # plt.subplot(num_queries, num_similar_images + 1, (num_similar_images + 1) * q + 1) 
    # OR ax=plt.subplot(3,4,q+1)
    # plt.imshow(img)
    # plt.annotate(f'Query {q}', (100,0))#:.7f
    # plt.axis('off')

    similar_images = ress['PerResNet18']['results'][query_name]
    distances = ress['PerResNet18']['distances'][query_name][0]
    for s,simimg_name in enumerate(similar_images):
        simimg_path = os.path.join(gallery_path + simimg_name)
        simimg = PIL.Image.open(simimg_path).convert('RGB')

        
        axes[q, s+1].imshow(simimg)
        #axes[q, s+1].annotate(f'd: {distances[s]}', (100,0))
        axes[q, s+1].text(0.5, 1.05, f'd: {distances[s]} ', ha='center', va='bottom', transform=axes[q, s+1].transAxes, fontsize= 8,fontstretch='ultra-condensed')
        axes[q, s+1].axis('off')

        # Alternative...
        #plt.subplot(num_queries, num_similar_images + 1, (num_similar_images + 1) * q + s + 2)
        # plt.imshow(simimg)
        # plt.annotate(f'd: {distances[s]}', (100,0))
        # plt.axis('off')

plt.show()


