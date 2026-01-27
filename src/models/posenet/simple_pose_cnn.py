from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import lovely_tensors as lt
from pytorch_model_summary import summary as psummary
from torchsummary import summary as tsummary


class SimplePoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(SimplePoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

    def from_pretrained(self, weights_path, device='cpu'):
        loaded_dict_dec = torch.load(weights_path, map_location=device)
        filtered_dict_dec = {k: v for k, v in loaded_dict_dec.items() if k in self.state_dict()}
        self.load_state_dict(filtered_dict_dec)
        self.eval()

if __name__=="__main__":
    lt.monkey_patch()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model=SimplePoseCNN(num_input_frames=2).to(device)
    x1=torch.rand(1, 3, 640, 192).to(device)
    x2=torch.rand(1, 3, 640, 192).to(device)
    input=torch.cat((x1, x2), dim=1)

    ######## 3 WAYS OF VISUALIZING THE ARCHITECTURE ########
    #architecture = psummary(model, input, max_depth=4, show_parent_layers=True, print_summary=True)
    tsummary(model, (6, 640, 192)) # USE WITHOUT BATCH DIMENSION, IT AUTOMATICALLY PUT -1 FOR IT
    # print(model)

    output=model(input)
    print("axisangle:", output[0].shape)
    print("translation:", output[1].shape)