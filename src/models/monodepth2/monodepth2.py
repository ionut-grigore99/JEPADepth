import torch
import torch.nn as nn
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt

from src.models.posenet.resnet_pose_cnn import ResnetEncoder
from src.models.monodepth2.depth_decoder import DepthDecoder

class MonoDepth2(nn.Module):
    def __init__(self, num_layers, pretrained, scales):
        super(MonoDepth2, self).__init__()
        self.encoder = ResnetEncoder(num_layers, pretrained, num_input_images=1)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=scales)

    def forward(self, x):
        features = self.encoder(x)
        disparity = self.decoder(features)

        return disparity

    def from_pretrained(self, encoder_weights_path, decoder_weights_path, device='cpu'):
        self.encoder.from_pretrained(encoder_weights_path, device)
        self.decoder.from_pretrained(decoder_weights_path, device)

if __name__ == "__main__":

    lt.monkey_patch()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_layers=18
    pretrained=True
    scales=[0, 1, 2, 3]

    model=MonoDepth2(num_layers, pretrained, scales).to(device)
    model.from_pretrained(
        encoder_weights_path="weights/monodepth2/mono_640x192/encoder.pth",
        decoder_weights_path="weights/monodepth2/mono_640x192/depth.pth",
        device=device
    )
    input=torch.rand(1, 3, 192, 640).to(device)

    ######## 2 WAYS OF VISUALIZING THE ARCHITECTURE ########
    architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output=model(input)
    print("disparity:", output[("disp", 0)].shape)
