import torch
import torch.nn as nn
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt

from src.models.monovit.hr_decoder import DepthDecoder
from src.models.monovit.mpvit import mpvit_small

class MonoViT(nn.Module):
    def __init__(self):
        super(MonoViT, self).__init__()
        self.encoder = mpvit_small()
        self.encoder.num_ch_enc = [64,128,216,288,288]
        self.decoder = DepthDecoder()

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

    model=MonoViT().to(device)
    model.from_pretrained(
        encoder_weights_path="weights/monovit/encoder.pth",
        decoder_weights_path="weights/monovit/depth.pth",
        device=device
    )
    input=torch.rand(1, 3, 192, 640).to(device)

    ######## 2 WAYS OF VISUALIZING THE ARCHITECTURE ########
    # architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output=model(input)
    print("disparity:", output[("disp", 0)].shape)
