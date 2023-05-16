import torchvision
import torchvision.transforms as T
import torch
import os
import PIL
import random

class MSE_Dataset(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

        self.target_transformation = T.Compose([
            T.RandomPerspective(0.3),
            T.RandomCrop(size=(180,180)),
            T.RandomHorizontalFlip(p=0.7),
            #T.RandomGrayscale(p=0.2),
            #T.RandomInvert(p=0.2),
            T.ColorJitter(brightness=.5, contrast=.3), 
            T.Resize((228,228), antialias=None),           
            ])


    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = PIL.Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, self.target_transformation(tensor_image)
    
class Triplet_Dataset(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset from folder, returning three tensor images for triplet loss. Anchor, positive and negative.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

        self.target_transformation = T.Compose([
            T.RandomPerspective(0.3),
            T.RandomCrop(size=(180,180)),
            T.RandomHorizontalFlip(p=0.7),
            #T.RandomGrayscale(p=0.2),
            #T.RandomInvert(p=0.2),
            T.ColorJitter(brightness=.5, contrast=.3), 
            T.Resize((228,228), antialias=None),           
            ])


    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = PIL.Image.open(img_loc).convert("RGB")

        for i in range(10): #it stops after 10 tries
            neg_idx = random.randint(0, len(self)-1)
            neg_loc = os.path.join(self.main_dir, self.all_imgs[neg_idx])
            neg_img = PIL.Image.open(neg_loc).convert("RGB")

            if not self.compare_img(self.all_imgs[idx], self.all_imgs[neg_idx]):
                break

        if self.transform is not None:
            tensor_image = self.transform(image)
            neg_tensor = self.transform(neg_img)

        return tensor_image, self.target_transformation(tensor_image), neg_tensor
    
    def compare_img(self, lbl1: str, lbl2: str):
        split1 = lbl1.split("_")
        split2 = lbl2.split("_")

        age1 = int(split1[0])
        age2 = int(split2[0])

        sex1 = split1[1]
        sex2 = split2[1]

        et1 = split1[2]
        et2 = split2[2]

        if abs(age1-age2) < 5 or sex1==sex2 or et1==et2:
            return True
        return False



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.img_names = os.listdir(dir_path)

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        if type(idx) == int:
            image_name = self.img_names[idx]
        elif type(idx) == str:
            image_name = [tmp for tmp in self.img_names if tmp == idx]
            image_name = image_name[0]
        else:
            print("ERROR")
            return

        image_path = os.path.join(self.dir_path, image_name)
        image = PIL.Image.open(image_path).convert("RGB")

        (width, height) = image.size

        if width > height:
            image = T.Pad((0,int((width-height)/2)), fill=[100,100,100], padding_mode="constant")(image)
        elif width < height:
            image = T.Pad((int((height-width)/2),0), fill=[100,100,100], padding_mode="constant")(image)

        if self.transform:
            image = self.transform(image)

        return image
    
def get_train_dataset(config, loss_function, transform):
    if loss_function == "MSE":
        full_dataset = MSE_Dataset(config["data_root"], transform)
    elif loss_function == "triplet":
        full_dataset = Triplet_Dataset(config["data_root"], transform)
    #train_loader = torch.utils.data.DataLoader(full_dataset, batch_size = config["batch_size"], shuffle=False)

    num_samples = len(full_dataset)
    training_samples = int(num_samples * 0.7 + 1)
    validation_samples = num_samples - training_samples
    training_data, validation_data = torch.utils.data.random_split(full_dataset,
                                                                   [training_samples, validation_samples])
    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(training_data, config["batch_size"], shuffle=True)#, num_workers=4)
    val_loader = torch.utils.data.DataLoader(validation_data, config["batch_size"], shuffle=False)#, num_workers=4)

    return full_dataset, train_loader, val_loader

def get_test_dataset(config, transform):
    test_dataset = TestDataset(config["data_root"], transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config["batch_size"], shuffle=False)

    return test_dataset, test_loader