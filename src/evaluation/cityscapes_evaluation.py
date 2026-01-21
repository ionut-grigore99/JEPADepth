import os
import sys
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import skimage
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import readlines
import tqdm
import matplotlib.pyplot as plt

from src.utils import disp_to_depth, readlines, count_parameters
from src.datasets.cityscapes_evaldataset import CityscapesEvalDataset
from src.config.conf import MambaDepthEnc_320x1024_Conf

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


current_file_path = os.path.dirname(__file__)
up_two_levels = os.path.join(current_file_path, '..')
splits_dir = os.path.join(up_two_levels, 'data', 'cityscapes/cityscapes_splits')
splits_dir = os.path.normpath(splits_dir)

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4



def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp



def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]

    assert get('evaluation_mode') in ["mono", "stereo"], "Please choose mono or stereo evaluation by setting evaluation_mode to 'mono' or 'stereo'!"

    if get('numpy_disparities_to_evaluate') is None: # optional path to a .npy disparities file to evaluate.

        # assert os.path.isdir(get('pretrained_models_folder')), "Cannot find a folder at {}".format(get('pretrained_models_folder'))

        # print("-> Loading weights from {}".format(get('pretrained_models_folder'))

        # Setup dataloaders
        filenames = readlines(os.path.join(splits_dir, get('evaluation_split'), "test_files.txt"))


        HEIGHT, WIDTH = conf.get('im_sz')[0], conf.get('im_sz')[1]
        

        dataset = datasets.CityscapesEvalDataset("/data/disertatie/cityscapes", filenames,
                                                HEIGHT, WIDTH,
                                                frames_to_load, 4,
                                                is_train=False)


        dataloader = DataLoader(dataset, 4, shuffle=False, num_workers=8,
                                pin_memory=True, drop_last=False)


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


        pred_disps = []
        src_imgs = []
        error_maps = []


        print('loading cityscapes gt depths individually due to their combined size!')
        gt_depths = os.path.join(splits_dir, opt.eval_split, "gt_depths")


        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                if torch.cuda.is_available():
                    input_color = input_color.cuda()
                    # print(input_color.shape, "==") # [16, 3, 192, 512]

                
                if get('evaluation_post_process'):
                    # print("post_process *********")
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = model(input_color)[0]

                pred_disp, _ = disp_to_depth(output, get('min_depth'), get('max_depth'))
                pred_disp = pred_disp.cpu()[:, 0].numpy()


                if get('evaluation_post_process'):
                    n = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:n], pred_disp[n:, :, ::-1])

                pred_disps.append(pred_disp)

                    

        pred_disps = np.concatenate(pred_disps)

        print('finished predicting!')


    else: # I don't care this
        # Load predictions from file
        print("-> Loading predictions from {}".format(get('numpy_disparities_to_evaluate')))
        pred_disps = np.load(get('numpy_disparities_to_evaluate'))

        if get('evaluate_eigen_to_benchmark'):
            eigen_to_benchmark_ids = np.load(os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))
            #IDK where to find eigen_to_benchmark_ids.npy file!.
            pred_disps = pred_disps[eigen_to_benchmark_ids]


    print("-> Evaluating")

    if get('evaluation_mode')=="stereo":
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        disable_median_scaling = True
        prediction_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        gt_depth = np.load(os.path.join(gt_depths, str(i).zfill(3) + '_depth.npy'))
        gt_height, gt_width = gt_depth.shape[:2]
        # crop ground truth to remove ego car -> this has happened in the dataloader for input
        # images
        gt_height = int(round(gt_height * 0.75))
        gt_depth = gt_depth[:gt_height]



        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # pred_depth = pred_disp
        pred_depth = 1 / pred_disp


        # when evaluating cityscapes, we centre crop to the middle 50% of the image.
        # Bottom 25% has already been removed - so crop the sides and the top here
        gt_depth = gt_depth[256:, 192:1856]
        pred_depth = pred_depth[256:, 192:1856]
        # 768, 2048
        # mono_depth = mono_depth[256:, 192:1856]


        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        
        if not disable_median_scaling:
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            ratios.append(ratio)
            pred_depth *= ratio

        
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        
        error_map = np.abs(gt_depth - pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        error_map = np.multiply(error_map, mask)
        # error_maps.append(error_map)

        errors.append(compute_errors(gt_depth, pred_depth))

    if get('save_predicted_disparities'):
        print("saving errors")
        if True:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            get('pretrained_models_folder'), "{}_{}_errors.npy".format(tag, get('evaluation_split')))
        np.save(output_path, np.array(errors))

    if not disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    if get('save_predicted_disparities'):
        # error_maps = np.concatenate(error_maps) # should not concatenate
        error_map_path = os.path.join(
            get('pretrained_models_folder'), "error_{}_split".format(get('evaluation_split')))
        print("-> Saving error maps to ", error_map_path)
        # np.save(error_map_path, error_maps)
        # np.savez_compressed(error_map_path, data=np.array(error_maps, dtype="object"))
    mean_errors = np.array(errors).mean(0)
    print(mean_errors)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":

    lt.monkey_patch()

    conf = MambaDepthEnc_320x1024_Conf().conf  # MambaDepthEnc_320x1024_Conf, MambaDepthBot_320x1024_Conf, ResNet50_320x1024_Conf, ResNet50_192x640_Conf, ConvNeXtLarge_320x1024_Conf, Effb5_320x1024_Conf
    get = lambda x: conf.get(x)

    evaluate(conf)

