from model_selection import *
from trainer import *


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Step 1: Define evaluation metrics
def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    return correct / total

# Step 2: Split your data

# Assuming you have already prepared your dataset and split it into train, validation, and test sets

train_dataset = trainer.train(train_loader= train_dataset)
val_dataset = trainer.train(val_loader=val_dataset)
test_dataset = trainer.train(test_loader=test_dataset)

# Step 3: Define model architectures

# Assuming you have defined multiple models with different architectures

model_1 = CNNencoder(config["model"])
model_2 = ResNet18(config["model"])
model_3 = PersonalizedVGG16_BN(config["model"])

# Step 4: Train the models

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Assuming you have defined the hyperparameters and optimizer

num_epochs = 1
learning_rate = config['training']['learning_rate']
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training model 1
model_1 = model_1.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_model(model_1, train_loader, criterion, optimizer, num_epochs)

# Training model 2
model_2 = model_2.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr=learning_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_model(model_2, train_loader, criterion, optimizer, num_epochs)

# Training model 3
model_3 = model_3.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_3.parameters(), lr=learning_rate)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_model(model_3, train_loader, criterion, optimizer, num_epochs)

# Step 5: Evaluate model performance

def evaluate_model(model, dataloader):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_accuracy += accuracy(outputs, labels)

    loss = running_loss / len(dataloader)
    accuracy = running_accuracy / len(dataloader)
    return loss, accuracy

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_1_loss, model_1_accuracy = evaluate_model(model_1, val_loader)
model_2_loss, model_2_accuracy = evaluate_model(model_2, val_loader)
model_3_loss, model_3_accuracy = evaluate_model(model_3, val_loader)

print("Model 1 - Loss: {:.4f}, Accuracy: {:.4f}".format(model_1_loss, model_1_accuracy))
print("Model 2 - Loss: {:.4f}, Accuracy: {:.4f}".format(model_2_loss, model_2_accuracy))
print("Model 3 - Loss: {:.4f}, Accuracy: {:.4f}".format(model_3_loss, model_3_accuracy))

# Step 6: Select the best model

best_model = model_1
best_accuracy = model_1_accuracy

if model_2_accuracy > best_accuracy:
    best_model = model_2
    best_accuracy = model_2_accuracy

if model_3_accuracy > best_accuracy:
    best_model = model_3
    best_accuracy = model_3_accuracy

print("Best Model - Loss: {:.4f}, Accuracy: {:.4f}".format(best_accuracy, best_accuracy))

# Step 7: Validate on the test set

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss, test_accuracy = evaluate_model(best_model, test_loader)

print("Test Set - Loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, test_accuracy))
