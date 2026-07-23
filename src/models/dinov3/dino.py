import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary
import lovely_tensors as lt

from src.models.dinov3.hub import dinov3_vits16, dinov3_vitb16, dinov3_vitl16
from src.models.dinov3.fpn_decoder import FPNDecoder
from src.models.pixio.dpt import DPTHead


class DINODepth(nn.Module):
    def __init__(self, encoder_size, decoder_channels, scales, decoder_type):
        super(DINODepth, self).__init__()
        self.num_scales   = len(scales)
        self.scales       = scales
        self.decoder_type = decoder_type

        model_map = {"small": dinov3_vits16, "base": dinov3_vitb16, "large": dinov3_vitl16}
        self.encoder = model_map[encoder_size](pretrained=False)
        weights_path = "weights/dino/dinov3_vits16.pth"  # Adjust path based on encoder_size if needed
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state_dict)

        if decoder_type == "fpn":
            self.decoder = FPNDecoder(in_channels=self.encoder.embed_dim, decoder_channels=decoder_channels, scales=scales)

        elif decoder_type == "dpt":
            self.decoder = DPTHead(
                nclass=1,
                in_channels=self.encoder.embed_dim,
                features=decoder_channels,
                out_channels=[256, 512, 1024, 1024],
                use_bn=False,
                scales=scales,
            )
        else:
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Choose 'fpn' or 'dpt'.")

    def forward(self, x):
        if self.decoder_type == "fpn":
            features = self.encoder.get_intermediate_layers(x, n=self.num_scales, reshape=True)
            disparity = self.decoder(features)
            return disparity

        else:  # dpt
            h, w = x.shape[-2:]
            features = self.encoder.get_intermediate_layers(x, n=self.num_scales, reshape=True)

            raw = self.decoder(features)

            # Interpolate to target sizes and apply sigmoid
            outputs = {}
            for s in self.scales:
                scale_factor = 1.0 / (2 ** s)
                target_h = int(h * scale_factor)
                target_w = int(w * scale_factor)
                outputs[("disp", s)] = torch.sigmoid(
                    F.interpolate(raw[("disp", s)], (target_h, target_w), mode="bilinear", align_corners=True)
                )
            return outputs

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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    encoder_size     = "small"
    decoder_channels = 256
    scales           = [0, 1, 2, 3]
    input            = torch.rand(1, 3, 192, 640).to(device)

    for decoder_type in ["fpn", "dpt"]:
        print(f"\n{'='*60}\n  decoder_type = '{decoder_type}'\n{'='*60}")
        model = DINODepth(encoder_size, decoder_channels, scales, decoder_type=decoder_type).to(device)
        output = model(input)
        for k in sorted(output.keys()):
            print(f"  {k}: {output[k].shape}")
