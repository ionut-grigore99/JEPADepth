import torch
import torch.nn as nn
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt

from src.models.dinov3.hub import dinov3_vits16, dinov3_vitb16, dinov3_vitl16
from src.models.dinov3.fpn_decoder import FPNDecoder

class DINODepth(nn.Module):
    def __init__(self, encoder_size, decoder_channels, scales):
        super(DINODepth, self).__init__()
        self.num_scales = len(scales)
        model_map = {"small": dinov3_vits16, "base": dinov3_vitb16, "large": dinov3_vitl16}
        self.encoder = model_map[encoder_size](pretrained=False)
        weights_path = "weights/dino/dinov3_vits16.pth"  # Adjust path based on encoder_size if needed
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)
        self.decoder = FPNDecoder(in_channels=self.encoder.embed_dim, decoder_channels=decoder_channels, scales=scales)

    def forward(self, x):
        features = self.encoder.get_intermediate_layers(x, n=self.num_scales, reshape=True)
        disparity = self.decoder(features)

        return disparity

    def from_pretrained(self, encoder_weights_path, decoder_weights_path, weights_path=None, device='cpu'):
        if weights_path is not None:
            # non-JEPA: full model saved as a single file
            loaded_dict_dec = torch.load(weights_path, map_location=device)
            filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
            self.load_state_dict(filtered_dict_dec)
            self.eval()
        else:
            # JEPA: encoder and decoder saved separately
            self.encoder.from_pretrained(encoder_weights_path, device)
            self.decoder.from_pretrained(decoder_weights_path, device)

if __name__ == "__main__":

    lt.monkey_patch()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    encoder_size="small"
    decoder_channels=256
    scales=[0, 1, 2, 3]

    model=DINODepth(encoder_size, decoder_channels, scales).to(device)
    # model.from_pretrained(
    #     encoder_weights_path="weights/jepa_small/encoder.pth",
    #     decoder_weights_path="weights/jepa_small/depth.pth",
    #     device=device
    # )
    input=torch.rand(1, 3, 192, 640).to(device)

    ######## 2 WAYS OF VISUALIZING THE ARCHITECTURE ########
    architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    # print(model)

    output=model(input)
    print("disparity:", output[("disp", 0)].shape)
