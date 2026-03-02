import torch
import torch.nn as nn
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt

from src.models.litemono.depth_encoder import LiteMono
from src.models.litemono.depth_decoder import DepthDecoder

class LiteMonoModel(nn.Module):
    def __init__(self, model_type, feed_height, feed_width, scales):
        super(LiteMonoModel, self).__init__()
        self.encoder = LiteMono(model=model_type, height=feed_height, width=feed_width)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=scales)

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

    feed_height = 192
    feed_width = 640
    model_type = "lite-mono" # "lite-mono", "lite-mono-8m"
    scales = range(3)

    model=LiteMonoModel(model_type, feed_height, feed_width, scales).to(device)
    model.from_pretrained(
        encoder_weights_path="weights/litemono/encoder.pth",
        decoder_weights_path="weights/litemono/depth.pth",
        device=device
    )
    input=torch.rand(1, 3, 192, 640).to(device)

    ######## 2 WAYS OF VISUALIZING THE ARCHITECTURE ########
    architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output=model(input)
    print("disparity:", output[("disp", 0)].shape)
