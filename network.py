import torchvision
import torch
import torch.nn as nn

class CNNencoder(nn.Module):
    """
    A simple Convolutional Encoder Model
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d((2, 2))

        self.lin1 = nn.Linear(256*7*7, 256*7*7) #n.features: 12544
        self.lin2 = nn.Linear(256*7*7, 256*7*7) #n.features: 12544
        self.lin3 = nn.Linear(256*7*7, 256*7)   #n.features: 1792

    def forward(self, x):
        # Downscale the image with conv maxpool etc.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        #x = self.flat(x)
        
        x = x.view(x.shape[0],-1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x

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

class ResNet50(torch.nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.resnet50(weights=weights)
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)


class PersonalizedResNet50(torch.nn.Module):
    def __init__(self, config):
        super(PersonalizedResNet50, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.resnet50(weights=weights)
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

class VGG11_BN(torch.nn.Module):
    def __init__(self, config):
        super(VGG11_BN, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.vgg11_bn(weights=weights)
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)

class PersonalizedVGG11_BN(torch.nn.Module):
    def __init__(self, config):
        super(PersonalizedVGG11_BN, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.vgg11_bn(weights=weights)
        # remove the last FC layer
        num_output_feats = self.net.classifier._modules['6'].in_features
        self.net.classifier._modules['6'] = torch.nn.Linear(num_output_feats, config["num_classes"])
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)

class VGG13_BN(torch.nn.Module):
    def __init__(self, config):
        super(VGG13_BN, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.vgg13_bn(weights=weights)
        
    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)

class PersonalizedVGG13_BN(torch.nn.Module):
    def __init__(self, config):
        super(PersonalizedVGG13_BN, self).__init__()
        # Load a resnet18 from torchvision, either use pretrained weights or not
        weights = "IMAGENET1K_V1" if config["pretrained"] else None
        self.net = torchvision.models.vgg13_bn(weights=weights)
        # remove the last FC layer
        num_output_feats = self.net.classifier._modules['6'].in_features
        self.net.classifier._modules['6'] = torch.nn.Linear(num_output_feats, config["num_classes"])

    def forward(self, x):
        logits = self.net(x)
        return logits

    # decorator, is used to tell pytorch to don't compute gradients when this function is called
    @torch.no_grad()
    def inference(self, x):
        return self.net(x)



