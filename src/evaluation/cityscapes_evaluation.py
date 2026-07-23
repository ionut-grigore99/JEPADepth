'''
Script to evaluate a depth estimation model on the Cityscapes dataset. 
Code taken from: https://github.com/nianticlabs/manydepth/blob/master/manydepth/evaluate_depth.py
@NOTE: link for downloading Cityscapes depth maps: https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip
'''

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import lovely_tensors as lt
import time

from src.models.pixio.dpt import DPTDepth
from src.models.diffnet.diffnet import DiffNet
from src.models.cadepth.cadepth import CADepth
from src.models.litemono.litemono import LiteMonoModel
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.models.dinov3.dino import DINODepth
from src.utils import disp_to_depth, readlines, count_parameters
from src.datasets.cityscapes_dataset import CityscapesDataset
from src.evaluation.utils import compute_errors, batch_post_process_disparity, STEREO_SCALE_FACTOR, create_evaluation_html_report
from src.config.conf import Conf


def evaluate(conf):
    """
        Evaluates a pretrained model using a specified test set.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if conf['numpy_disparities_to_evaluate'] is not None: # optional path to load disparities from a .npy file for evaluation - no model inference.
        print("-> Loading predictions from {}".format(conf['numpy_disparities_to_evaluate']))
        pred_disps = np.load(conf['numpy_disparities_to_evaluate'])
    else: # perform model inference to get disparities for evaluation.
        # Loading Cityscapes dataset
        filenames = readlines(os.path.join(conf['cityscapes_dataset_path'], "cityscapes_test_files.txt"))
        dataset = CityscapesDataset(conf['cityscapes_dataset_path'], filenames, conf['im_sz'][0], conf['im_sz'][1], [0], 4, is_train=False, img_ext='.png' if conf['train_from_png'] else '.jpg')
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=conf['num_workers'], pin_memory=True, drop_last=False)
        #@NOTE: - drop_last=True ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size)
        #       - drop_last=False will make the last batch smaller than your batch_size.

        # Preparing model
        if conf['model_name'].startswith("pixio"):
            model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'], scales=conf['pixio']['scales'])
            model.from_pretrained(weights_path=conf['pixio']['weights_path'], device=device)
        elif conf['model_name'] == "dino":
            model = DINODepth(conf['dino']['encoder_size'], conf['dino']['decoder_channels'], scales=conf['dino']['scales'], decoder_type=conf['dino']['decoder_type'])
            model.from_pretrained(encoder_weights_path=conf['dino']['encoder_weights_path'], decoder_weights_path=conf['dino']['decoder_weights_path'], weights_path=conf['dino']['weights_path'], device=device)
        elif conf['model_name'] == "litemono":
            model = LiteMonoModel(conf['litemono']['model_type'], conf['im_sz'][0], conf['im_sz'][1], scales=conf['litemono']['scales'])
            model.from_pretrained(encoder_weights_path=conf['litemono']['encoder_weights_path'], decoder_weights_path=conf['litemono']['decoder_weights_path'], device=device)
        elif conf['model_name'] == "diffnet":
            model = DiffNet(scales=conf['diffnet']['scales'])
            model.from_pretrained(encoder_weights_path=conf['diffnet']['encoder_weights_path'], decoder_weights_path=conf['diffnet']['decoder_weights_path'], device=device)
        elif conf['model_name'] == "monovit":
            from src.models.monovit.monovit import MonoViT
            model = MonoViT()
            model.from_pretrained(encoder_weights_path=conf['monovit']['encoder_weights_path'], decoder_weights_path=conf['monovit']['decoder_weights_path'], device=device)
        elif conf['model_name'] == "cadepth":
            model = CADepth(num_layers=conf['cadepth']['num_layers'])
            model.from_pretrained(encoder_weights_path=conf['cadepth']['encoder_weights_path'], decoder_weights_path=conf['cadepth']['decoder_weights_path'], device=device)
        elif conf['model_name'] == "monodepth2":
            model = MonoDepth2(num_layers=conf['monodepth2']['num_layers'], pretrained=conf['monodepth2']['pretrained'], scales=conf['monodepth2']['scales'])
            model.from_pretrained(encoder_weights_path=conf['monodepth2']['encoder_weights_path'], decoder_weights_path=conf['monodepth2']['decoder_weights_path'], device=device)
        else:
            raise NotImplementedError("Model not implemented for evaluation!")
        model.to(device)
        model.eval()
        
        # Count model parameters
        total_params, _ = count_parameters(model)

        pred_disps = []
        src_imgs = []
        error_maps = []
        inference_times = []

        print("\n-> Computing predictions with size {}x{} (WxH) using {}".format(conf['im_sz'][1], conf['im_sz'][0], conf['model_name']))

        step = 0
        with torch.no_grad():
            for data in tqdm(dataloader):
                step = step + 1
                input_color = data[("color", 0, 0)].to(device) # tensor[1, 3, H, W] with values between 0 and 1!

                if conf['evaluation_post_process']: # flipping post processing from the Monodepthv1 paper! (https://arxiv.org/pdf/1609.03677 - pg. 5 bottom right)
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0) # tensor[2, 3, H, W] with values between 0 and 1!

                # Measure inference time
                start_time = time.time()
                normalized_disparity = model(input_color)[("disp", 0)]
                pred_disp, _ = disp_to_depth(normalized_disparity, conf['min_depth'], conf['max_depth'])
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                
                # Synchronize CUDA if using GPU to get accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                if conf['evaluation_post_process']: # flipping post processing from the Monodepthv1 paper! (https://arxiv.org/pdf/1609.03677 - pg. 5 bottom right)
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1]) # pred_disp.shape will be now (1, H, W)

                pred_disps.append(pred_disp) 
                src_imgs.append(data[("color", 0, 0)])

        pred_disps = np.concatenate(pred_disps)
        src_imgs = np.concatenate(src_imgs)

    gt_depths = os.path.join(conf['cityscapes_dataset_path'], "gt_depths")
    print("-> Evaluating")

    disable_median_scaling=conf['disable_median_scaling'] #@NOTE: median scaling is described in Section 4.1 from https://arxiv.org/pdf/1704.07813
    prediction_depth_scale_factor=conf['prediction_depth_scale_factor']
    if conf['evaluation_mode']=="stereo":
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        disable_median_scaling = True
        prediction_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    sample_images = []
    max_samples = conf['html_max_samples']  # Number of sample images to include in HTML report
    
    for i in range(pred_disps.shape[0]):
        gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
        gt_height, gt_width = gt_depth.shape[:2]
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        # when evaluating cityscapes, we centre crop to the middle 50% of the image.
        # Bottom 25% has already been removed - so crop the sides and the top here
        gt_depth = gt_depth[256:, 192:1856]
        pred_depth = pred_depth[256:, 192:1856]
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        error_maps.append(error_map)

        pred_depth *= prediction_depth_scale_factor # prediction_depth_scale_factor is 1 in case of mono.
        if not disable_median_scaling: # apply median scaling as described in Section 4.1 from https://arxiv.org/pdf/1704.07813
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

        # Collect sample images for HTML report (every N images to get good distribution)
        if len(sample_images) < max_samples and i % max(1, pred_disps.shape[0] // max_samples) == 0:
            # Get the original input image
            if conf['numpy_disparities_to_evaluate'] is None:  # We have access to the dataset
                # Load ground truth depth (full, before cropping)
                gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
                gt_height, gt_width = gt_depth.shape[:2]
                gt_height = int(round(gt_height * 0.75))
                gt_depth = gt_depth[:gt_height]
                
                # Reconstruct full depth and disparity maps (before center crop) for visualization
                pred_disp = pred_disps[i]
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1 / pred_disp
                
                # Apply the same scaling that was used for evaluation
                pred_depth *= prediction_depth_scale_factor
                if not disable_median_scaling and len(ratios) > 0:
                    pred_depth *= ratios[-1]
                pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
                
                # Get filename from dataset
                filename = filenames[i]
                
                # Get input image (original size before any cropping)
                input_img = dataset[i][("color", 0, 0)]  
                input_img = input_img.permute(1, 2, 0).cpu().numpy()
                input_img = (input_img * 255).astype(np.uint8)
                
                sample_images.append({
                    'filename': filename,
                    'input_img': input_img,
                    'pred_depth': pred_depth,
                    'gt_depth': gt_depth,
                    'pred_disp': pred_disp
                })
    
    mean_errors = np.array(errors).mean(0)
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times) if len(inference_times) > 0 else 0.0

    print("\n| " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("|{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "|")
    print(f"\n-> Average inference time: {avg_inference_time:.3f}s per image")

    # Generate HTML evaluation report
    create_evaluation_html_report("cityscapes", mean_errors, conf, len(dataloader), sample_images=sample_images if len(sample_images) > 0 else None, num_parameters=total_params, avg_inference_time=avg_inference_time)
    
    print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = Conf().conf
    evaluate(conf)