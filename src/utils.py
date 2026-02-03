import math
import torch
from io import BytesIO
import base64
import PIL.Image as pil
from ptflops import get_model_complexity_info


def readlines(filename):
    """
        Read all the lines in a text file and return them as a list.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def normalize_image(im):
    """
        Rescale image pixels to span range [0, 1].
    """
    maximum = float(torch.max(im).cpu().item())
    minimum = float(torch.min(im).cpu().item())
    d = maximum - minimum if maximum != minimum else 1e5
    return (im - minimum) / d

def disp_to_depth(disp, min_depth, max_depth):
    """
        Convert network's sigmoid output_images into depth prediction.
        The formula for this conversion is given in the 'additional considerations' section of the paper:
            "We convert the sigmoid output σ to depth with D = 1/(aσ + b), where a and b are chosen to constrain D between 0.1 and 100 units."
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def sec_to_hm(t):
    """
        Convert time in seconds to time in hours, minutes and seconds.
        e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """
        Convert time in seconds to a nice string.
        e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

def rot_from_axisangle(vec):
    """
        Convert an axisangle rotation into a 4x4 transformation matrix. (adapted from https://github.com/Wallacoloo/printipi)
        Input 'vec' has to be B x 1 x 3.
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """
        Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

def transformation_from_parameters(axisangle, translation, invert=False):
    """
        Convert the network's (axisangle, translation) output_images into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def predict_poses(conf, models, inputs): 
    """
        Predict poses between input frames for monocular sequences.
    """
    outputs = {}

    num_input_frames = len(conf['frame_ids_training'])
    num_pose_frames = 2 if conf['pose_model_input'] == "pairs" else num_input_frames # in our case we'll have num_pose_frames=2
    if num_pose_frames == 2: # In our case we will go through this branch.
        # In this setting, we compute the pose to each source frame via a separate forward pass through the pose network.
        # select what features the pose network takes as input
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in conf['frame_ids_training']} # for each frame it will take the input as scale=0
        for f_i in conf['frame_ids_training'][1:]:
            if f_i != "s":
                # To maintain ordering we always pass frames in temporal order
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                axisangle, translation = models["pose_model"](torch.cat(pose_inputs, 1))
                outputs[("axisangle", 0, f_i)] = axisangle 
                outputs[("translation", 0, f_i)] = translation 

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    else:
        # Here we input all frames to the pose net (and predict all poses) together
        pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in conf['frame_ids_training'] if i != "s"], 1)
        axisangle, translation = models["pose_model"](pose_inputs)
        for i, f_i in enumerate(conf['frame_ids_training'][1:]):
            if f_i != "s":
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

    return outputs

def count_parameters(model):
    """
        Return both total parameters and total trainable parameters.
    """
    total_params = 0
    total_trainable = 0

    for _, parameter in model.named_parameters():
        num = parameter.numel()
        total_params += num
        if parameter.requires_grad:
            total_trainable += num

    return total_params, total_trainable

def get_complexity(model, resolution):
    """
        Get the computational complexity and number of parameters of a model."""
    macs, params = get_model_complexity_info(model, (3, resolution[0], resolution[1]), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
        Fills the input Tensor with values drawn from a truncated normal distribution. 
    """
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
        Fills the input Tensor with values drawn from a truncated normal distribution.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def apply_masks(x, masks):
    """
        Apply a list of masks to the input tensor and return the masked patches concatenated along the batch dimension.
        Args:
            :x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
            :masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def repeat_interleave_batch(x, B, repeat):
    """
        Repeat the input tensor along the batch dimension by <repeat> times.
    """
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)
    
def image_to_base64(image):
    """Convert PIL Image to base64 string for HTML embedding"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def numpy_to_base64(array):
    """Convert numpy array to base64 string"""
    im = pil.fromarray(array)
    return image_to_base64(im)

def format_model_name(model_name):
    """Format model name for display (e.g., 'monodepth2' -> 'MonoDepth2')"""
    model_name_map = {
        'monodepth2': 'MonoDepth2',
        'pixio': 'Pixio',
        'pixio_vitb16': 'Pixio ViT-B/16',
        'pixio_vitl16': 'Pixio ViT-L/16',
        'pixio_vith16': 'Pixio ViT-H/16',
    }
    return model_name_map.get(model_name, model_name.title())