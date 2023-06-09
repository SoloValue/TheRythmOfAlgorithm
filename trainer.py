import torch
from sklearn.neighbors import NearestNeighbors

from dataset import *
from network import *
import utils

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
            # in this case we store only the model
            torch.save(self.model.state_dict(), f'{self.config["model_path"]}best.pth') 
        else:
            save_dict = dict(
                model = self.model.state_dict(),
                # optimizer has parameters as well, you want to save this to be able to go back to this exact stage of training
                optimizer = self.optimizer.state_dict(), 
            )
            torch.save(save_dict, f'{self.checkpoint_path}epoch-{epoch}.pth')

    def train_one_epoch(self, data_loader):
        samples = 0.
        cumulative_loss = 0.

        self.model.train()
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

                # === Apply the loss == #
                #Triplet loss
                loss = self.cost_function(anc_enc, pos_enc, neg_enc) 
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

            #add the number of images in the batch (anc and pos are batches)
            samples += anc.shape[0] 
            cumulative_loss += loss.item()

        return cumulative_loss / samples

    def validation_step(self, data_loader):
        samples = 0.
        cumulative_loss = 0.

        self.model.eval() 
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
                #add the number of images in the batch (anc and pos are batches)
                samples += anc.shape[0] 
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

            labels = {
                "1" : [],
                "2" : []
            }

            for i, img_name in enumerate(img_names):
                if img_name[0] == target_letter:
                    labels["1"].append(i)
                else:
                    labels["2"].append(i)
                    
            # sum of correct matches among top-k ranking
            ck = 0     
            for i in index_list:
                if i in labels["1"]:
                    ck += 1
            
            
            if len(labels["1"]) > 0:
                test_error = 1-(ck / len(labels["1"]))
            else:
                test_error = -1

            tot_test_error += test_error

        return distance_list, indices_list, tot_test_error/len(target_letters)

    def comp_step(self, query_loader: TestLoader, gallery_loader: TestLoader, top_n: int):
        """ 
        Takes query, gallery and number of top-k to rank.
        Returns:
        -  distance_list : list of distance of i-th gallery-image from query;
        -  indices_list : list of image indexes corresponding in distances_list;
        """

        self.model.eval()

        query_names = query_loader.img_names
        gallery_names = gallery_loader.img_names

        embedding = None
        target_embedding = None

        with torch.no_grad():
            # create embedding (gallery)
            for i,image in enumerate(gallery_loader):
                image = image.cuda()
                image_enc = self.model(image).cpu()

                if i==0: 
                    embedding = image_enc
                else:
                    embedding = torch.cat((embedding, image_enc), 0)
            
            # create embedding (query)
            for i,image in enumerate(query_loader):
                image = image.cuda()
                image_enc = self.model(image).cpu()

                if i==0: 
                    target_embedding = image_enc
                else:
                    target_embedding = torch.cat((target_embedding, image_enc), 0)

            # ranking
            embedding = embedding.cpu().detach().numpy()
            target_embedding = target_embedding.cpu().detach().numpy()

            knn = NearestNeighbors(n_neighbors=top_n, metric="cosine")
            knn.fit(embedding)
            results = dict()
            distances = dict()
            for q,query_name in enumerate(query_names):
                #print(target_embedding.shape())
                #print(target_embedding[q].shape())
                distance_list, indices_list = knn.kneighbors(target_embedding[q].reshape(1, -1), return_distance=True)
                indices_list = indices_list.tolist()
                distance_list = distance_list.tolist()    
                index_list = indices_list[0]

                results[query_name] = [gallery_names[index_img] for index_img in index_list]
                distances[query_name] = distance_list

        return results, distances        # results are for submitting, distances for plotting purposes

    def train(self, train_loader, val_loader, test_loader):

        # For each epoch, train the network and then compute evaluation results
        best_test_loss = 9999.9
        print(f"\tStart training...")

        for e in range(self.max_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validation_step(val_loader)
            _,_,test_loss = self.test_step(test_loader)
            print('======= Epoch: {:d}'.format(e))
            print(f'\t Training loss: {train_loss:.5f}')
            print(f'\t Validation loss: {val_loss:.5f}')
            print(f'\t Test loss: {test_loss}')

            utils.log_run(train_loss, val_loss, test_loss)

            # == Save the model checkpoints == #
            
            # if the current epoch is in the interval, or is the last epoch -> save
            if e % self.save_checkpoint_every == 0 or e == (self.max_epochs - 1):  
                self.save(e, is_best=False)                

            # Early Stopping
            if test_loss > best_test_loss:
                self.trigger += 1
                if self.trigger == self.early_stopping_patience:
                    print(f"Validation Accuracy did not improve for {self.early_stopping_patience} epochs. Killing the training...")
                    break
            else:
                # update the best val loss so far
                self.save(e, is_best=True)
                print("\t...saving best model...")
                best_test_loss = test_loss
                self.trigger = 0
            print('-----------------------------------------------------')
            # ===========================================
        print(f"\t...end of training")
    
