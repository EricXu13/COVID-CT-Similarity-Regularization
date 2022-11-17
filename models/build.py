import torch.nn as nn
from torchvision import models
from .covidnet import CovidNet


def build_model(config):
    if config.MODEL.ARCH == 'covidnet':
        model = CovidNet(num_classes=config.MODEL.NUM_CLASSES)
    elif hasattr(models, config.MODEL.ARCH):
        if config.MODEL.ARCH.lower().__contains__('inception'):
            model = getattr(models, config.MODEL.ARCH)(num_classes=config.MODEL.NUM_CLASSES, aux_logits=False)
        else:
            model = getattr(models, config.MODEL.ARCH)(num_classes=config.MODEL.NUM_CLASSES)
    else:
        raise NotImplementedError(f"Unkown model: {config.MODEL.ARCH}")
        
    return model
        
    