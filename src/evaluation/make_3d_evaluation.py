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

from src.utils import disp_to_depth, readlines, count_parameters
from src.config.conf import MambaDepthEnc_320x1024_Conf

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

    main_path="/data/disertatie/make3d/test"

    with open(os.path.join(main_path, "make3d_test_files.txt")) as f:
        test_filenames = f.read().splitlines()


    if conf.get('model_name').startswith("mambaDepth"):
        model = get_mambaDepth_enc_model(conf)
        model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/mambaDepthEnc_320x1024/20240205-230952/models/weights_epoch_14/depth_encoder_decoder.pth', device='cuda:6')
        #model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/mambaDepthBot_320x1024/20240206-214227/models/weights_epoch_19/depth_encoder_decoder.pth', device='cuda')
        #model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/mambaDepthEnc_320x1024/20240207-180812/models/weights_epoch_19/depth_encoder_decoder.pth', device='cuda')
        #model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/mambaDepthEnc_320x1024/20240209-191013/models/weights_epoch_22/depth_encoder_decoder.pth', device='cuda')
        #model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/mambaDepthEnc_320x1024/20240210-220346/models/weights_epoch_16/depth_encoder_decoder.pth', device='cuda')
    elif conf.get('model_name').startswith("monodepth"):
        model = MonoDepth2EncoderDecoder()
        #model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/monodepth2/20240213-130024/models/weights_epoch_13/depth_encoder_decoder.pth', device='cuda:6')
        model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/monodepth2/20240215-091943/models/weights_epoch_6/depth_encoder_decoder.pth', device='cuda:6')
    elif conf.get('model_name').startswith('vmunet'):
        model = VMUNet(
                        num_classes=1,
                        input_channels=3,
                        depths=[2,2,2,2],
                        depths_decoder=[2,2,2,1],
                        drop_path_rate=0.2,
                        load_ckpt_path='/data/disertatie/MambaDepth/src/pretrained/vmamba_small_e238_ema.pth'
                        )
        model.load_from()
        model.from_pretrained(weights_path='/data/disertatie/MambaDepth/src/tensorboard/train/vmunet/20240214-224651/models/weights_epoch_13/depth_encoder_decoder.pth', device='cuda:6')

    model.to('cuda:6')
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




        # breakpoint()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # by me
        # image = image.astype('float32') / 255.0    # by me

        # input_color = torch.from_numpy(image)   # by me
        # input_color = input_color.permute(2, 0, 1).unsqueeze(0).to('cuda:6')   # by me

        # breakpoint()

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

    conf = MambaDepthEnc_320x1024_Conf().conf  # MambaDepthEnc_320x1024_Conf, MambaDepthBot_320x1024_Conf, ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf
    get = lambda x: conf.get(x)

    evaluate(conf)

