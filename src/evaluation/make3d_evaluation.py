'''
Script to evaluate a depth estimation model on the Make3D dataset. Code taken from: https://github.com/nianticlabs/monodepth2/issues/392
NOTE: In order to check the correctness of this evaluation code, please compare its results to those obtained in the Monodepth2 paper (page 7, Table 3).
Expected results when evaluating Monodepth2 model with 640x192 pretrained weights are:

|  abs_rel |   sq_rel |     rmse | rmse_log | 
|    0.321 |    3.377 |    7.252 |    0.163 | 
@NOTE: The results may slightly vary depending on the environment and library versions.
'''


import os
import cv2
import numpy as np
import torch
import lovely_tensors as lt
from scipy import io
from tqdm import tqdm

from src.models.pixio.dpt import DPTDepth
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.utils import disp_to_depth
from src.config.conf import Conf

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


def compute_errors(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log



def evaluate(conf):
    """
        Evaluates a pretrained model using the 134 test images from Make3D.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    MIN_DEPTH = 0
    MAX_DEPTH = 70

    # Preparing model
    if conf['model_name'].startswith("pixio"):
        model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'], scales=conf['pixio']['scales'])
        model.from_pretrained(weights_path=conf['pixio']['weights_path'], device=device)
    elif conf['model_name'] == "monodepth2":
        model = MonoDepth2(num_layers=conf['monodepth2']['num_layers'], pretrained=conf['monodepth2']['pretrained'], scales=conf['monodepth2']['scales'])
        model.from_pretrained(encoder_weights_path=conf['monodepth2']['encoder_weights_path'], decoder_weights_path=conf['monodepth2']['decoder_weights_path'], device=device)
    else:
        raise NotImplementedError("Model not implemented for evaluation!")
    model.to(device)
    model.eval()

    with open(os.path.join(conf['make3d_dataset_path'], "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()
    test_filenames = map(lambda x: x[4:], test_filenames)
    

    depths_gt = []
    images = []
    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = int(1704 / 2)
    depth_new_height = 21
    for filename in test_filenames:
        mat = io.loadmat(os.path.join(conf['make3d_dataset_path'], "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
        depths_gt.append(mat["Position3DGrid"][:,:,3])

        image = cv2.imread(os.path.join(conf['make3d_dataset_path'], "Test134", "img-{}.jpg".format(filename)))
        image = image[ int((2272 - color_new_height)/2):int((2272 + color_new_height)/2),:,:]
        images.append(image[:,:,::-1])
    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = map(lambda x: x[int((55 - 21)/2):int((55 + 21)/2),:], depths_gt)

    depths_gt_cropped = list(depths_gt_cropped)
    errors = []
    print("\n-> Computing predictions with size {}x{} (WxH) using {}".format(conf['im_sz'][1], conf['im_sz'][0], conf['model_name']))
    with torch.no_grad():
        for i in tqdm(range(len( images))):
            input_color = images[i]
            input_color = cv2.resize(input_color/255.0, (conf['im_sz'][1], conf['im_sz'][0]), interpolation=cv2.INTER_NEAREST)            
            input_color = torch.tensor(input_color, dtype = torch.float).cuda().permute(2,0,1)[None,:,:,:]
            output = model(input_color)
            pred_disp, _ = disp_to_depth(output[("disp", 0)], conf['min_depth'], conf['max_depth'])
            pred_disp = pred_disp.squeeze().cpu().numpy()
            depth_gt = depths_gt_cropped[i]
            depth_pred = 1 / pred_disp
            depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(depth_gt > MIN_DEPTH, depth_gt < MAX_DEPTH)
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= np.median(depth_gt) / np.median(depth_pred)
            depth_pred[depth_pred > MAX_DEPTH] = MAX_DEPTH
            errors.append(compute_errors(depth_gt, depth_pred))
        mean_errors = np.mean(errors, 0)

    print(("\n| " + "{:>8} | " * 4).format( "abs_rel", "sq_rel", "rmse", "rmse_log"))
    print(("| "+"{: 8.3f} | " * 4).format(*mean_errors.tolist()))

    print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = Conf().conf
    evaluate(conf)

