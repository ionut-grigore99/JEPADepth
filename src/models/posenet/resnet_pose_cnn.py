import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from collections import OrderedDict
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt


class ResNetMultiImageInput(models.ResNet):
    """Constructs a ResNet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of ResNet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = torch.utils.model_zoo.load_url({18: ResNet18_Weights.IMAGENET1K_V1, 50: ResNet50_Weights.IMAGENET1K_V1}[num_layers].url)
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a ResNet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

    
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):


        last_features = [f[-1] for f in input_features]


        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]


        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

class ResNetPoseCNN(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images, num_input_features, num_frames_to_predict_for, stride=1):
        super(ResNetPoseCNN, self).__init__()
        self.encoder = ResnetEncoder(num_layers, pretrained, num_input_images)
        self.decoder = PoseDecoder(num_ch_enc=self.encoder.num_ch_enc, num_input_features=num_input_features, num_frames_to_predict_for=num_frames_to_predict_for, stride=stride)

    def forward(self, x):
        features = self.encoder(x)
        axisangle, translation = self.decoder([features])
        
        return axisangle, translation 

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_dec = torch.load(weights_path, map_location=device)
        filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_dec)
        self.eval()


if __name__=="__main__":

    lt.monkey_patch()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_layers=18
    pretrained=True
    num_input_images=2  # Number of images concatenated as input to encoder
    num_input_features=1  # Number of feature pyramids (1 since concatenated before encoding)
    num_frames_to_predict_for=2  # Number of relative poses (bidirectional: 0->1 and 1->0)

    model=ResNetPoseCNN(num_layers, pretrained, num_input_images, num_input_features, num_frames_to_predict_for).to(device)
    x1=torch.rand(1, 3, 640, 192).to(device)
    x2=torch.rand(1, 3, 640, 192).to(device)
    input=torch.cat((x1, x2), dim=1)

    ######## 2 WAYS OF VISUALIZING THE ARCHITECTURE ########
    architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output=model(input)
    print("axisangle:", output[0].shape)
    print("translation:", output[1].shape)

