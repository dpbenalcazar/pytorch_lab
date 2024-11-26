import torch
import torch.nn as nn
from torchvision import models
from argparse import ArgumentParser as argparse

def print_layers(model):
    children_counter = 0
    for n, c in model.named_children():
        print("Children Counter: ",children_counter, " Layer Name: ", n,)
        children_counter+=1
    return

def load_model(backbone, weights="None", classes=3):
    if backbone == "alexnet":
        if weights == "imagenet":
            model = models.alexnet(weights="IMAGENET1K_V1")
        else:
            model = models.alexnet()
        n_inputs = model.classifier[6].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[6] = last_layer
    
    elif backbone == "densenet121":
        if weights == "imagenet":
            model = models.densenet121(weights="IMAGENET1K_V1")
        else:
            model = models.densenet121()
        n_inputs = model.classifier.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier = last_layer

    elif backbone == "densenet161":
        if weights == "imagenet":
            model = models.densenet161(weights="IMAGENET1K_V1")
        else:
            model = models.densenet161()
        n_inputs = model.classifier.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier = last_layer

    
    elif backbone == "densenet169":
        if weights == "imagenet":
            model = models.densenet169(weights="IMAGENET1K_V1")
        else:
            model = models.densenet169()
        n_inputs = model.classifier.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier = last_layer
    
    elif backbone == "densenet201":
        if weights == "imagenet":
            model = models.densenet201(weights="IMAGENET1K_V1")
        else:
            model = models.densenet201()
        n_inputs = model.classifier.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier = last_layer

    elif backbone == "efficientnet_v2_l":
        if weights == "imagenet":
            model = models.efficientnet_v2_l(weights="DEFAULT")
        else:
            model = models.efficientnet_v2_l()
        n_inputs = model.classifier[1].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[1] = last_layer

    elif backbone == "efficientnet_v2_m":
        if weights == "imagenet":
            model = models.efficientnet_v2_m(weights="DEFAULT")
        else:
            model = models.efficientnet_v2_m()
        n_inputs = model.classifier[1].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[1] = last_layer

    elif backbone == "efficientnet_v2_s":
        if weights == "imagenet":
            model = models.efficientnet_v2_s(weights="DEFAULT")
        else:
            model = models.efficientnet_v2_s()
        n_inputs = model.classifier[1].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[1] = last_layer

    elif backbone == "inception_v3":
        if weights == "imagenet":
            model = models.inception_v3(weights="DEFAULT")
        else:
            model = models.inception_v3()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer

    elif backbone == "maxvit_t":
        if weights == "imagenet":
            model = models.maxvit_t(weights="IMAGENET1K_V1")
        else:
            model = models.maxvit_t()
        n_inputs = model.classifier[5].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[5] = last_layer

    elif backbone == "mobilenet_v2":
        if weights == "imagenet":
            model = models.mobilenet_v2(weights="IMAGENET1K_V2")
        else:
            model = models.mobilenet_v2()
        n_inputs = model.classifier[1].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[1] = last_layer

    elif backbone == "mobilenet_v3_small":
        if weights == "imagenet":
            model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        else:
            model = models.mobilenet_v3_small()
        n_inputs = model.classifier[3].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[3] = last_layer

    elif backbone == "mobilenet_v3_large":
        if weights == "imagenet":
            model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")
        else:
            model = models.mobilenet_v3_large()
        n_inputs = model.classifier[3].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[3] = last_layer

    elif backbone == "resnet18":
        if weights == "imagenet":
            model = models.resnet18(weights="IMAGENET1K_V1")
        else:
            model = models.resnet18()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer

    elif backbone == "resnet34":
        if weights == "imagenet":
            model = models.resnet34(weights="IMAGENET1K_V1")
        else:
            model = models.resnet34()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer

    elif backbone == "resnet50":
        if weights == "imagenet":
            model = models.resnet50(weights="IMAGENET1K_V2")
        else:
            model = models.resnet50()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer
            
    elif backbone == "resnet101":
        if weights == "imagenet":
            model = models.resnet101(weights="IMAGENET1K_V2")
        else:
            model = models.resnet101()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer

    elif backbone == "resnet152":
        if weights == "imagenet":
            model = models.resnet152(weights="IMAGENET1K_V2")
        else:
            model = models.resnet152()
        n_inputs = model.fc.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.fc = last_layer

    elif backbone == "swin_v2_t":
        if weights == "imagenet":
            model = models.swin_v2_t(weights="IMAGENET1K_V1")
        else:
            model = models.swin_v2_t()
        n_inputs = model.head.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.head = last_layer

    elif backbone == "swin_v2_s":
        if weights == "imagenet":
            model = models.swin_v2_s(weights="IMAGENET1K_V1")
        else:
            model = models.swin_v2_s()
        n_inputs = model.head.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.head = last_layer

    elif backbone == "swin_v2_b":
        if weights == "imagenet":
            model = models.swin_v2_b(weights="IMAGENET1K_V1")
        else:
            model = models.swin_v2_b()
        n_inputs = model.head.in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.head = last_layer

    elif backbone == "vgg16":
        if weights == "imagenet":
            model = models.vgg16(weights="IMAGENET1K_V1")
        else:
            model = models.vgg16()
        n_inputs = model.classifier[6].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[6] = last_layer

    elif backbone == "vgg19":
        if weights == "imagenet":
            model = models.vgg19(weights="IMAGENET1K_V1")
        else:
            model = models.vgg19()
        n_inputs = model.classifier[6].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.classifier[6] = last_layer

    elif backbone == "vit_b_16":
        if weights == "imagenet":
            model = models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
        else:
            model = models.vit_b_16()
        n_inputs = model.heads[0].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.heads[0] = last_layer

    elif backbone == "vit_b_32":
        if weights == "imagenet":
            model = models.vit_b_32(weights="IMAGENET1K_V1")
        else:
            model = models.vit_b_32()
        n_inputs = model.heads[0].in_features
        last_layer = nn.Sequential(
            nn.Linear(n_inputs, classes),
            nn.Sigmoid(),
        )
        model.heads[0] = last_layer
        
    return model

if __name__ == '__main__':
    parser = argparse()
    parser.add_argument('-b', '--backbone', type=str, default="vgg16",
                        help='Conv-Net backbone.')
    parser.add_argument('-w', '--weights', type=str, default="none",
                        help="Model's initial Weights: < none | imagenet | /path/to/weights/ >")
    parser.add_argument('-c', '--classes', type=int, default=5,
                        help='Number of output classes.')
    args = parser.parse_args()

    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\nWorking with: {}".format(device))

    # Get pretrained model
    model = load_model(args.backbone, args.weights, args.classes)
    model.to(device)
    print('{} model loaded on {} with weights: {}'.format(args.backbone, device, args.weights))

    print('\nLayers:')
    print_layers(model)