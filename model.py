import torchvision.models as models
import torch.nn as nn

def construct_encoder():
    
    model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    
    return model