from .alexnet import *
from .vgg import *
from .lenet import *
from .zfnet import *
from .resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .squeezenet import *
from .squeezenetv2 import *
from .squeezenetv3 import *


def Backbone(input_shape, classes, filters = 64, include_top = True, backbone_name = "VGG11"):
    if   backbone_name == "VGG11"       : model = VGG(classes, "vgg11", filters, include_top)
    elif backbone_name == "VGG13"       : model = VGG(classes, "vgg13", filters, include_top)
    elif backbone_name == "VGG16"       : model = VGG(classes, "vgg16", filters, include_top)
    elif backbone_name == "VGG19"       : model = VGG(classes, "vgg19", filters, include_top)
    elif backbone_name == "alexnet"     : model = Alexnet(classes, filters, include_top)
    elif backbone_name == "lenet"       : model = LeNet(classes, include_top)
    elif backbone_name == "zfnet"       : model = ZFnet(classes, filters, include_top)
    elif backbone_name == "resnet18"    : model = Resnet_1(classes, "resnet18", filters, include_top)
    elif backbone_name == "resnet34"    : model = Resnet_1(classes, "resnet34", filters, include_top)
    elif backbone_name == "resnet50"    : model = Resnet_2(classes, "resnet50", filters, include_top)
    elif backbone_name == "resnet101"   : model = Resnet_2(classes, "resnet101", filters, include_top)
    elif backbone_name == "resnet152"   : model = Resnet_2(classes, "resnet152", filters, include_top)
    elif backbone_name == "mobilenet"   : model = Mobilenet(classes, filters, include_top)
    elif backbone_name == "mobilenetv2" : model = Mobilenetv2(classes, filters, include_top)
    elif backbone_name == "squeezenet"  : model = Squeezenet(classes, filters, include_top)
    elif backbone_name == "squeezenetv2": model = Squeezenetv2(classes, filters, include_top)
    elif backbone_name == "squeezenetv3": model = Squeezenetv3(classes, filters, include_top)

    model.build(input_shape = input_shape)
    return model
