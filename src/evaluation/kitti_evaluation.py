'''
Script to evaluate a depth estimation model on the KITTI dataset.
NOTE: In order to check the correctness of this evaluation code, please compare its results to those obtained in the Monodepth2 paper (Table 1 - page 6 and Table 7 - page 14).
Expected results when evaluating Monodepth2 model with 1024x320 pretrained weights (using the 'eigen' split, monocular mode, without post-processing, with median scaling) are:

|  abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
|   0.115  |   0.884  |   4.700  |   0.190  |   0.879  |   0.961  |   0.982  |

Expected results when evaluating Monodepth2 model with 640x192 pretrained weights (using the 'eigen_benchmark' split, monocular mode, without post-processing, with median scaling) are:

|  abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
|   0.090  |   0.545  |   3.945  |   0.137  |   0.914  |   0.983  |   0.995  |

@NOTE: The results may slightly vary depending on the environment and library versions.
@NOTE: The reported metrics correspond to evaluating each model at the same resolution it was trained on.
'''


import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import lovely_tensors as lt
from torch.utils.data import DataLoader
import time

from src.models.pixio.dpt import DPTDepth
from src.models.litemono.litemono import LiteMonoModel
from src.models.diffnet.diffnet import DiffNet
from src.models.cadepth.cadepth import CADepth
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.models.dinov3.dino import DINODepth
from src.utils import disp_to_depth, readlines, count_parameters
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.evaluation.utils import compute_errors, batch_post_process_disparity, create_evaluation_html_report, STEREO_SCALE_FACTOR
from src.config.conf import Conf

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def evaluate(conf):
    """
        Evaluates a pretrained model using a specified test set.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    splits_dir = conf['data_path'].replace('kitti_data', 'kitti_splits')

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if conf['numpy_disparities_to_evaluate'] is not None: # optional path to load disparities from a .npy file for evaluation - no model inference.
        print("-> Loading predictions from {}".format(conf['numpy_disparities_to_evaluate']))
        pred_disps = np.load(conf['numpy_disparities_to_evaluate'])

        if conf['evaluate_eigen_to_benchmark']:
            eigen_to_benchmark_ids = np.load(os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]
    else: # perform model inference to get disparities for evaluation.
        # Loading KITTI dataset
        filenames = readlines(os.path.join(splits_dir, conf['evaluation_split'], "test_files.txt"))
        dataset = KITTIRAWDataset(conf['data_path'], filenames, conf['im_sz'][0], conf['im_sz'][1], [0], 4, is_train=False, img_ext='.png' if conf['train_from_png'] else '.jpg')
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=conf['num_workers'], pin_memory=True, drop_last=False)
        #@NOTE: - drop_last=True ignores the last batch (when the number of examples in your dataset is not divisible by your batch_size)
        #       - drop_last=False will make the last batch smaller than your batch_size.

        # Preparing model
        if conf['model_name'].startswith("pixio"):
            model = DPTDepth(conf['pixio']['encoder'], conf['pixio']['pretrained_ckp'], conf['pixio']['scales'])
            model.from_pretrained(weights_path=conf['pixio']['weights_path'], device=device)
        elif conf['model_name'] == "dino":
            model = DINODepth(conf['dino']['encoder_size'], conf['dino']['decoder_channels'], scales=conf['dino']['scales'], decoder_type=conf['dino']['decoder_type'])
            model.from_pretrained(encoder_weights_path=conf['dino']['encoder_weights_path'], decoder_weights_path=conf['dino']['decoder_weights_path'], weights_path=conf['dino']['weights_path'], device=device)
        elif conf['model_name'] == "litemono":
            model = LiteMonoModel(model_type=conf['litemono']['model_type'], feed_height=conf['im_sz'][0], feed_width=conf['im_sz'][1], scales=conf['litemono']['scales'])
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

    if conf['evaluation_split'] == 'benchmark': # This is the official KITTI benchmark set - when you want to submit results to the online server.
        save_dir = os.path.join(conf['load_weights_folder'], "benchmark_predictions")
        print("-> Saving out KITTI benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized # This is the conversion to metric-scale depth assuming a nominal baseline of 0.1 units.
            depth = np.clip(depth, 0, 80) # This is because the KITTI benchmark requires depth maps to be in the range [0, 80]
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, conf['evaluation_split'], "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
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
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if conf['evaluation_split'] == "eigen": 
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0
        
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
                # Reconstruct unmasked depth and disparity for visualization
                pred_disp = pred_disps[i]
                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1 / pred_disp

                # Get unmasked ground truth depth
                gt_depth = gt_depths[i]
                
                # Get filename 
                filename = filenames[i]
                
                # Get input image
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
    avg_inference_time = sum(inference_times) / len(inference_times)

    print("\n| " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("|{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "|")
    print(f"\n-> Average inference time: {avg_inference_time:.3f}s per image")
    
    # Generate HTML evaluation report
    create_evaluation_html_report("kitti", mean_errors, conf, len(dataloader), sample_images=sample_images if len(sample_images) > 0 else None, num_parameters=total_params, avg_inference_time=avg_inference_time)

    print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = Conf().conf
    evaluate(conf)


