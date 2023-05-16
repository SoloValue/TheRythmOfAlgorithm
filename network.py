import torchvision
import torch

class ResNet18(torch.nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.resnet18(weights=weights)
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)


class PersonalizedResNet18(torch.nn.Module):
    def __init__(self, config):
        super(PersonalizedResNet18, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.resnet18(weights=weights)
        # remove the last FC layer
        num_output_feats = self.net.fc.in_features   # dim  of the features
        self.net.fc = torch.nn.Linear(num_output_feats, config["num_classes"])  # Initialize a new fully connected layer, with num_output = num_classes
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)
    
class PersonalizedVGG16_BN(torch.nn.Module):
    def __init__(self, config):
        super(PersonalizedVGG16_BN, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.vgg16_bn(weights=weights)
        # remove the last FC layer
        num_output_feats = self.net.fc.in_features   # dim  of the features
        self.net.fc = torch.nn.Linear(num_output_feats, config["num_classes"])  # Initialize a new fully connected layer, with num_output = num_classes
