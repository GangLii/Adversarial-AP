import torch
import torchvision
import logging
import sys


def resnet18(num_classes=1):
    model = torchvision.models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    # model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    return model

def resnet34(num_classes=1):
    model = torchvision.models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    
    return model

def resnet50(num_classes=1):
    model = torchvision.models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)

    return model


def logger_init(out_pth):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(stream_handler)

    # FileHandler
    file_handler = logging.FileHandler(out_pth)
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('logging')
    
    return logger



