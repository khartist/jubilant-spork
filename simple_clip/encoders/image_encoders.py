import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
import torch
import timm
from pprint import pprint






"""
Encode Image to a fixed size vector 
using MobileNetV3Small or TinyVit5M
"""
class ImageEncoder(nn.Module):

    class Squeeze(nn.Module):
        def forward(self, x):
            return x.squeeze(-1).squeeze(-1)


    def _prep_encoder(self, model):
        modules = list(model.children())[:-1]
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(self.Squeeze())

        return nn.Sequential(*modules)
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small
    def mobile_net_v3_small(self):
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        return self._prep_encoder(model)

    # https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k
    def tiny_vit_5m(self):
        model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True)
        model.reset_classifier(0)

        return self._prep_encoder(model) 

    def __init__(self, model_name):
        super(ImageEncoder, self).__init__()
        if model_name == "mobile_net_v3_small":
            self.model = self.mobile_net_v3_small()
        elif model_name == "tiny_vit_5m":
            self.model = self.tiny_vit_5m()
        else:
            raise ValueError(f"Model {model_name} not found")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    model = ImageEncoder("mobile_net_v3_small")
    print("Model loaded successfully")
    img = torch.rand(1, 3, 224, 224)
    print(model(img).shape)

    model = ImageEncoder("tiny_vit_5m")
    print("Model loaded successfully")
    img = torch.rand(1, 3, 224, 224)
    print(model(img).shape)