import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, pad: int = 0):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class FPNDecoder(nn.Module):
    """
    FPN-like top-down decoder that returns Monodepth2-style multi-scale disparity:
      ("disp", 0): [B, 1, H,   W]
      ("disp", 1): [B, 1, H/2, W/2]
      ("disp", 2): [B, 1, H/4, W/4]
      ("disp", 3): [B, 1, H/8, W/8]

    Assumption: encoder features are at /16 resolution (ViT patch grid).
    We upsample progressively: /16 -> /8 -> /4 -> /2 -> /1.
    """

    def __init__(
        self,
        in_channels: int,
        decoder_channels: int,
        scales: tuple[int, ...],
    ):
        super().__init__()
        self.scales = tuple(scales)

        # Lateral projections (one per encoder feature you pass in)
        # If your features list has 4 tensors, we use 4 laterals.
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels, decoder_channels, kernel_size=1) for _ in range(4)
        ])

        # Refinement convs after fusion at each stage (keep spatial size)
        self.refine = nn.ModuleList([
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, pad=1),
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, pad=1),
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, pad=1),
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, pad=1),
        ])

        # Prediction heads (one per requested scale)
        self.disp_head = nn.ModuleDict({
            str(s): nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1) # Predicting 1 channel for disparity
            for s in self.scales
        })

    def _resize_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, features: list[torch.Tensor], input_hw: tuple[int, int] | None = None):
        """
        features: list of 4 feature maps, typically from ViT intermediate layers.
                  Each: [B, C, H16, W16] (usually same H/W for all in ViT)
        input_hw: (H, W) of original image. If None, we infer H/W assuming features are /16.
        """
        assert len(features) >= 4, f"Expected >=4 feature maps, got {len(features)}"
        feats = features[:4]
       
        # Infer input size if not provided
        if input_hw is None:
            h16, w16 = feats[-1].shape[-2:]
            H, W = h16 * 16, w16 * 16
        else:
            H, W = input_hw

        # Create laterals
        # We interpret feats[0] as the "shallowest" among the chosen layers and feats[-1] deepest.
        lat = [proj(f) for proj, f in zip(self.lateral, feats)]

        # Start from deepest
        x = lat[-1]  # /16

        outputs = {}

        # We will produce stages:
        # stage at /8  -> scale 3
        # stage at /4  -> scale 2
        # stage at /2  -> scale 1
        # stage at /1  -> scale 0

        # 1) /16 -> /8 (scale 3)
        x = F.interpolate(x, scale_factor=2, mode="nearest")          # /8
        x = x + self._resize_to(lat[-2], x)                           # fuse
        x = self.refine[0](x)
        if 3 in self.scales:
            disp3 = torch.sigmoid(self.disp_head["3"](x))
            # make sure exact size is (H/8, W/8)
            disp3 = F.interpolate(disp3, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
            outputs[("disp", 3)] = disp3

        # 2) /8 -> /4 (scale 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")          # /4
        x = x + self._resize_to(lat[-3], x)
        x = self.refine[1](x)
        if 2 in self.scales:
            disp2 = torch.sigmoid(self.disp_head["2"](x))
            disp2 = F.interpolate(disp2, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
            outputs[("disp", 2)] = disp2

        # 3) /4 -> /2 (scale 1)
        x = F.interpolate(x, scale_factor=2, mode="nearest")          # /2
        x = x + self._resize_to(lat[-4], x)
        x = self.refine[2](x)
        if 1 in self.scales:
            disp1 = torch.sigmoid(self.disp_head["1"](x))
            disp1 = F.interpolate(disp1, size=(H // 2, W // 2), mode="bilinear", align_corners=False)
            outputs[("disp", 1)] = disp1

        # 4) /2 -> /1 (scale 0)
        x = F.interpolate(x, scale_factor=2, mode="nearest")          # /1
        x = self.refine[3](x)
        if 0 in self.scales:
            disp0 = torch.sigmoid(self.disp_head["0"](x))
            disp0 = F.interpolate(disp0, size=(H, W), mode="bilinear", align_corners=False)
            outputs[("disp", 0)] = disp0

        return outputs
    
    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_dec = torch.load(weights_path, map_location=device)
        filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_dec)
        self.eval()


if __name__ == "__main__":
    from src.models.dinov3.hub import dinov3_vits16
    dino_encoder = dinov3_vits16(pretrained=False)

    imgs = torch.randn(2, 3, 192, 640)
    
    # Features at different scales (from intermediate layers)
    features = dino_encoder.get_intermediate_layers(imgs, n=4, reshape=True) # Get 4 intermediate layers, reshaped to (B, C, H/16, W/16)
    
    # Create decoder
    decoder = FPNDecoder(
        in_channels=dino_encoder.embed_dim,
        decoder_channels=256,
        scales=[0, 1, 2, 3]
    )
    
    # Forward pass
    outputs = decoder(features)
    
    # Print output shapes (Monodepth2 style)
    print("Multi-scale disparity outputs:")
    for scale in [0, 1, 2, 3]:
        if ("disp", scale) in outputs:
            disp = outputs[("disp", scale)]
            print(f"  Scale {scale}: {disp.shape}")
    
    # Expected output shapes:
    # Scale 0: [2, 1, 192, 640] (H, W)  
    # Scale 1: [2, 1, 96, 320]  (H/2, W/2)
    # Scale 2: [2, 1, 48, 160]  (H/4, W/4)
    # Scale 3: [2, 1, 24, 80]   (H/8, W/8)
