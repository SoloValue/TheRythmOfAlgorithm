import wandb
import torch
from sklearn.neighbors import NearestNeighbors

from dataset import *
from network import *

class UnsupervisedTransferLearnTrainer:
    def __init__(self, model: torch.nn.Module, config):
        self.config = config
        self.model = model

        if config["loss_function"] == "MSE":
            self.cost_function = torch.nn.MSELoss()
        elif self.config["loss_function"] == "triplet":
            self.cost_function = torch.nn.TripletMarginLoss()

        # Parameters
        self.max_epochs = config["max_epochs"]
        self.save_checkpoint_every = config["save_checkpoint_every"]
        self.early_stopping_patience = config["early_stopping_patience"]
        self.checkpoint_path = config["model_path"] + "checkpoints/"
        self.trigger = 0
    
    def SetupTrain(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

    def save(self, epoch, is_best=False):
        if is_best:
            torch.save(self.model.state_dict(), f'{self.config["model_path"]}best.pth') # in this case we store only the model
        else:
            save_dict = dict(
                model = self.model.state_dict(),
                optimizer = self.optimizer.state_dict(), # optimizer has parameters as well, you want to save this to be able to go back to this exact stage of training
            )
            torch.save(save_dict, f'{self.checkpoint_path}epoch-{epoch}.pth')

    def train_one_epoch(self, data_loader):
        samples = 0.
        cumulative_loss = 0.

        self.model.train()  # Strictly needed if network contains layers which has different behaviours between train and test
        for batch_idx, image_vector in enumerate(data_loader):
            
            if self.config["loss_function"] == "triplet":
                # Load data into GPU
                anc = image_vector[0].cuda()
                pos = image_vector[1].cuda()
                neg = image_vector[2].cuda()

                # Forward pass (encoding the images)
                anc_enc = self.model(anc)
                pos_enc = self.model(pos)
                neg_enc = self.model(neg)

                # Apply the loss
                loss = self.cost_function(anc_enc, pos_enc, neg_enc) #Triplet loss
            elif self.config["loss_function"] == "MSE":
                # Load data into GPU
                anc = image_vector[0].cuda()
                pos = image_vector[1].cuda()

                # Forward pass (encoding the images)
                anc_enc = self.model(anc)
                pos_enc = self.model(pos)

                # Apply the loss
                loss = self.cost_function(anc_enc, pos_enc) #MSE
            else:
                print("Choose a proper error function!")
                return -1

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Reset the gradients
            self.optimizer.zero_grad()

            # Better print something, no?
            samples += anc.shape[0] #add the number of images in the batch (!! anc and pos are batches !!)
            cumulative_loss += loss.item()

        return cumulative_loss / samples

    def validation_step(self, data_loader):
        samples = 0.
        cumulative_loss = 0.

        self.model.eval()  # Strictly needed if network contains layers which has different behaviours between train and test
        with torch.no_grad():
            for batch_idx, image_vector in enumerate(data_loader):
                if self.config["loss_function"] == "triplet":
                    # Load data into GPU
                    anc = image_vector[0].cuda()
                    pos = image_vector[1].cuda()
                    neg = image_vector[2].cuda()

                    # Forward pass (encoding the images)
                    anc_enc = self.model(anc)
                    pos_enc = self.model(pos)
                    neg_enc = self.model(neg)

                    # Apply the loss
                    loss = self.cost_function(anc_enc, pos_enc, neg_enc) #Triplet loss
                elif self.config["loss_function"] == "MSE":
                # Load data into GPU
                    anc = image_vector[0].cuda()
                    pos = image_vector[1].cuda()

                    # Forward pass (encoding the images)
                    anc_enc = self.model(anc)
                    pos_enc = self.model(pos)

                    # Apply the loss
                    loss = self.cost_function(anc_enc, pos_enc) #MSE
                else:
                    print("Choose a proper error function!")
                    return -1

                # Better print something, no?
                samples += anc.shape[0] #add the number of images in the batch (!! anc and pos are batches !!)
                cumulative_loss += loss.item()

        return cumulative_loss / samples
    
    def test_step(self, data_loader: TestLoader):
        self.model.eval()

        img_names = data_loader.img_names

        target_letters = ['a', 'b', 'c', 'd', 'e']
        tot_test_error = 0
        for target_letter in target_letters:
            test_error = 0
            target_index = [img_names.index(nome) for nome in img_names if nome[0]==target_letter and nome[1]=='0'][0]

            embedding = None
            target_embedding = None

            with torch.no_grad():
                for i,image in enumerate(data_loader):
                    image = image.cuda()
                    image_enc = self.model(image).cpu()
                    if i == target_index:
                        target_embedding = image_enc

                    if i==0: 
                        embedding = image_enc
                    else:
                        embedding = torch.cat((embedding, image_enc), 0)
            
            embedding = embedding.cpu().detach().numpy()
            target_embedding = target_embedding.cpu().detach().numpy()

            knn = NearestNeighbors(n_neighbors=5, metric="cosine")
            knn.fit(embedding)

            distance_list, indices_list = knn.kneighbors(target_embedding, return_distance=True)
            indices_list = indices_list.tolist()

            index_list = indices_list[0]

            ######################## ELISA & SARA #############################
            labels = {
                "1" : [],
                "2" : []
            }

            for i, img_name in enumerate(img_names):
                if img_name[0] == target_letter:
                    labels["1"].append(i)
                else:
                    labels["2"].append(i)
            
            ck = 0     # sum of correct amatches among top-k ranking
            for i in index_list:
                if i in labels["1"]:
                    ck += 1
                #else:
                #    break
            
            if len(labels["1"]) > 0:
                test_error = 1-(ck / len(labels["1"]))
            else:
                test_error = -1

            tot_test_error += test_error

        return distance_list, indices_list, tot_test_error/len(target_letters)

    def train(self, train_loader, val_loader, test_loader):

        # For each epoch, train the network and then compute evaluation results
        best_val_loss = 9999.9
        print(f"\tStart training...")

        for e in range(self.max_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validation_step(val_loader)
            _,_,test_loss = self.test_step(test_loader)
            print('======= Epoch: {:d}'.format(e))
            print(f'\t Training loss: {train_loss:.5f}')
            print(f'\t Validation loss: {val_loss:.5f}')
            print(f'\t Test loss: {test_loss}')

            wandb.log({
                "loss/train":train_loss,
                "loss/val":val_loss,
                "loss/test":test_loss,
            })

            # Save the model checkpoints
            if e % self.save_checkpoint_every == 0 or e == (self.max_epochs - 1):  # if the current epoch is in the interval, or is the last epoch -> save
                self.save(e, is_best=False)                

            # Early Stopping
            if val_loss > best_val_loss:
                self.trigger += 1
                if self.trigger == self.early_stopping_patience:
                    print(f"Validation Accuracy did not improve for {self.early_stopping_patience} epochs. Killing the training...")
                    break
            else:
                # update the best val loss so far
                self.save(e, is_best=True)
                print("\t...saving best model...")
                best_val_loss = val_loss
                self.trigger = 0
            print('-----------------------------------------------------')
            # ===========================================
        print(f"\t...end of training")
    
