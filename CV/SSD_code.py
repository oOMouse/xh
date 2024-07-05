import torch
import torch.nn as nn
import torchvision.models as models


def t0():
    net = models.detection.ssd300_vgg16(weights=None, weights_backbone=None)
    print(net)


if __name__ == '__main__':
    t0()
