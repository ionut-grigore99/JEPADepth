from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import scipy
import torch
from tqdm import tqdm
import lovely_tensors as lt
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_model_summary import summary as psummary

from src.models.pixio.pixio import DPTDepth
from src.utils import disp_to_depth, readlines, count_parameters
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

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(conf):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    main_path="/data/disertatie/make3d/test"

    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()


    if conf.get('model_name').startswith("pixio"):
        model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'], conf['pixio']['scales'])
        model.from_pretrained(weights_path='...', device=device)
    else:   
        raise NotImplementedError("Model not implemented for evaluation!")

    model.to(device)
    model.eval()

    depths_gt = []
    images = []
    ratio = 2
    h_ratio = 1 / (1.33333 * ratio)
    color_new_height = 1704 / 2
    depth_new_height = 21
    for filename in test_filenames:
        mat = scipy.io.loadmat(os.path.join(main_path, "Test134Depth", "Gridlaserdata", "depth_sph_corr-{}.mat".format(filename)))
        depths_gt.append(mat["Position3DGrid"][:,:,3])
        image = cv2.imread(os.path.join(main_path, "Test134Img", "img-{}.jpg".format(filename)))
        image = image[ int((2272 - color_new_height)/2):int((2272 + color_new_height)/2),:, :]
        images.append(image[:,:,::-1])

    depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    depths_gt_cropped = map(lambda x: x[int((55 - 21)/2):int((55 + 21)/2),:], depths_gt)

    depths_gt_cropped = list(depths_gt_cropped)
    errors = []
    with torch.no_grad():
        for i in range(len(images)):
            input_color = images[i]
            input_color = cv2.resize(input_color/255.0, (1024, 320), interpolation=cv2.INTER_AREA)
            
            input_color = torch.tensor(input_color, dtype = torch.float).to('cuda:6').permute(2,0,1)[None,:,:,:]

                            
            output = model(input_color)[0]
            pred_disp, _ = disp_to_depth(output, 1e-3, 60)
            pred_disp = pred_disp.squeeze().cpu().numpy()
            depth_gt = depths_gt_cropped[i]
            depth_pred = 1 / pred_disp
            depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
            mask = np.logical_and(depth_gt > 0, depth_gt < 60)
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= np.median(depth_gt) / np.median(depth_pred)
            depth_pred[depth_pred > 60] = 60
            errors.append(compute_errors(depth_gt, depth_pred))
        mean_errors = np.mean(errors, 0)

    print(("{:>8} | " * 4).format( "abs_rel", "sq_rel", "rmse", "rmse_log"))
    print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))
    print("\n-> Done!")




        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # by me
        # image = image.astype('float32') / 255.0    # by me

        # input_color = torch.from_numpy(image)   # by me
        # input_color = input_color.permute(2, 0, 1).unsqueeze(0).to('cuda:6')   # by me

        ######################################################## by me

        # if get('evaluation_post_process'): # for flipping post processing from the original Monodepth paper!
        #     # Post-processed results require each image to have two forward passes
        #     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0) # tensor[2, 3, 320, 1024] with values between 0 and 1!

        # output = model(input_color)[0]

        # pred_disp, _ = disp_to_depth(output, get('min_depth'), get('max_depth'))

        # pred_disp = pred_disp.cpu().numpy()

        # pred_disp = pred_disp.cpu()[:, 0].numpy() # if get('evaluation_post_process') is True, then pred_disp.shape is (2, 160, 512) <-> (2, H/2, W/2)

        # if get('evaluation_post_process'): # flipping post processing from the original Monodepth paper!
        #     N = pred_disp.shape[0] // 2
        #     pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1]) # pred_disp.shape will be now (1, 160, 512)

        ######################################################## by me
        #breakpoint()
        # pred_disps.append(pred_disp) 



    # depths_gt_resized = map(lambda x: cv2.resize(x, (305, 407), interpolation=cv2.INTER_NEAREST), depths_gt)
    # depths_gt_cropped = map(lambda x: x[(55 - 21)/2:(55 + 21)/2, :], depths_gt)


    # errors = []
    # for i in range(len(test_filenames)):
    #     depth_gt = depths_gt_cropped[i]
    #     depth_pred = 1 / pred_disps[i]
    #     depth_pred = cv2.resize(depth_pred, depth_gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    #     mask = np.logical_and(depth_gt > 0, depth_gt < 70)
    #     depth_gt = depth_gt[mask]
    #     depth_pred = depth_pred[mask]
    #     depth_pred *= np.median(depth_gt) / np.median(depth_pred)
    #     depth_pred[depth_pred > 70] = 70
    #     errors.append(compute_errors(depth_gt, depth_pred))
    # mean_errors = np.mean(errors, 0)

    # print(("{:>8} | " * 4).format( "abs_rel", "sq_rel", "rmse", "rmse_log"))
    # print(("{: 8.3f} , " * 4).format(*mean_errors.tolist()))
    # print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = Conf().conf
    evaluate(conf)

