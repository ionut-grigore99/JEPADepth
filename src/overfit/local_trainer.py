from __future__ import absolute_import, division, print_function
from json import encoder
from pathlib import Path

import lovely_tensors as lt
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_model_summary import summary

from src.config.conf import Conf
from src.models.posenet.simple_pose_cnn import SimplePoseCNN
from src.models.posenet.resnet_pose_cnn import ResNetPoseCNN 
from src.models.pixio.dpt import DPTDepth
from src.models.layers import *
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *


class Overfit:
    def __init__(self, conf):
        self.conf = conf
        self.log_path = os.path.join(self.conf['tensorboard_path'], 'overfit')
        self.log_path = os.path.join(self.log_path, self.conf['model_name'])
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_scales = len(self.conf['loss_scales']) # default=[0], we only perform single scale training => num_scales=1
        self.num_input_frames = len(self.conf['frame_ids_training']) # default=[0, -1, 1] => num_input_frames=3
        self.num_pose_frames = 2 if self.conf['pose_model_input'] == "pairs" else self.num_input_frames # default=2
        self.min_depth = self.conf['min_depth'] 
        self.max_depth = self.conf['max_depth'] 

        # Prepare depth model
        if self.conf['model_name'].startswith("pixio"):
            self.models["depth_model"] = DPTDepth(self.conf['pixio']['encoder'], self.conf['pixio']['pretrained_ckp'])
            self.models["depth_model"] = self.models["depth_model"].to(self.device)
        else:
            print("Model not recognized!")
            exit()
        if self.conf['load_pretrained_depth_model']:
            self.models["depth_model"].from_pretrained()
        self.models["depth_model"] = self.models["depth_model"].to(self.device)
        self.parameters_to_train += list(self.models["depth_model"].parameters())

        # Prepare pose model
        if self.conf['pose_model_type']=="simple_pose_cnn":
            self.models["pose_model"] = SimplePoseCNN(self.num_pose_frames)
        elif self.conf['pose_model_type']=="resnet_pose_cnn":
            self.models["pose_model"] = ResNetPoseCNN(self.conf['resnet_pose_cnn']['num_layers'], self.conf['resnet_pose_cnn']['pretrained'], self.conf['resnet_pose_cnn']['num_input_images'], self.conf['resnet_pose_cnn']['num_input_features'], self.conf['resnet_pose_cnn']['num_frames_to_predict_for'])
        else:
            print("Pose model type not recognized!")
            exit()
        if self.conf['load_pretrained_pose_model']:
            self.models["pose_model"].from_pretrained(weights_path=self.conf['pose_model_weights_path'], device=self.device)
        self.models["pose_model"] = self.models["pose_model"].to(self.device)
        self.parameters_to_train += list(self.models["pose_model"].parameters())


        self.model_optimizer = optim.Adam(self.parameters_to_train, self.conf['learning_rate'])

        print("Overfiting model named:\n  ", self.conf['model_name'])
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Overfiting is using:\n  ", self.device)
        print("Using split:\n  ", self.conf['training_split'])

        # Preparing data for training
        self.dataset = KITTIRAWDataset

        data_path = Path(self.conf["data_path"])
        overfit_filenames = readlines(data_path.parent / "kitti_splits" / self.conf['training_split'] / "overfit_files.txt")
        img_ext = '.png' if self.conf['train_from_png'] else '.jpg'

        overfit_dataset = self.dataset(self.conf['data_path'], overfit_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=True, img_ext=img_ext)
        self.overfit_loader = DataLoader(overfit_dataset, self.conf['bs'], False, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)

        self.batch = next(iter(self.overfit_loader))

        self.writers = {}
        self.writers['overfit'] = SummaryWriter(self.log_path)

        # Layer to compute the SSIM loss between a pair of images. We always use it.
        if self.conf['use_ssim']:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.conf['loss_scales']: # default we have just 1 scale which is set to 0!
            h = self.conf['im_sz'][0] // (2 ** scale) # this will do nothing basically
            w = self.conf['im_sz'][1] // (2 ** scale) # also this

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.conf['bs'], h, w)
            self.backproject_depth[scale].to(self.device)

            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.conf['bs'], h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        self.save_opts()

    def overfit_batch(self, iters=2000):
        self.step=0
        metrics = {}
        for i in (tbar := tqdm(range(iters), desc="Overfit")):
            self.model_optimizer.zero_grad()
            outputs_dict, losses = self.process_batch(self.batch) 
            if "depth_gt" in self.batch:
                compute_depth_metrics(self.batch, outputs_dict, metrics, self.depth_metric_names) 
            self.log("overfit", self.batch, outputs_dict, losses, metrics) #  Write to the tensorboard events file.
            losses["loss"].backward()
            if self.conf.get('clip_grad_norm'): torch.nn.utils.clip_grad_norm_(self.models["depth_encoder_decoder"].parameters(), 1)
            self.model_optimizer.step()
            self.step += 1
        print("Overfit done!")

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        disp_maps = self.models["depth_model"](inputs["color_aug", 0, 0]) # ["color_aug", 0, 0] means frame_id 0 and scale 0 (which is full resolution 320x1024)
                                                                          # the output of our depth_model is disparity which is 1/depth

        # Predict poses between input frames for monocular sequences.
        poses=predict_poses(conf, self.models, inputs)

        outputs_dict = self.generate_warped_images(inputs, disp_maps, poses) # See the diagram of self-supervised MDE pipeline. 
        # 'outputs_dict' dictionary contains the warped (reprojected) color images, the predicted depth_maps and the color input frame (interpolated at the original resolution if necessary).

        losses = compute_losses(self.conf, inputs, disp_maps, outputs_dict, self.ssim)

        return outputs_dict, losses

    def generate_warped_images(self, inputs, disp_maps, poses):
        """
            Generate the warped (reprojected) color images for a minibatch saved into the 'outputs_dict' dictionary.
            Apart from these, we also save in the 'outputs_dict' predicted disp_maps and also depth_maps (1/disp_maps) 
            interpolated at the original resolution.
            'outputs_dict':
                -the predicted disp_maps unscaled interpolated at the original resolution:  outputs_dict[("disp_unscaled", 0, scale)]
                -the predicted disp_maps scaled interpolated at the original resolution:  outputs_dict[("disp_scaled", 0, scale)]
                -the predicted depth_maps interpolated at the original resolution:  outputs_dict[("depth", 0, scale)]
                -the warped (reprojected) color images for a minibatch: outputs_dict[("color", frame_id, scale)]
        """
        outputs_dict={}
        for scale in self.conf['loss_scales']:
            if self.conf['model_name'].startswith("pixio"):
                disp = disp_maps
            else:
                disp = disp_maps[scale]

            if self.conf['monodepthv1_multiscale']:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.conf['im_sz'][0], self.conf['im_sz'][1]], mode="bicubic", align_corners=True)
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.conf['frame_ids_training'][1:]): # basically frame_ids_training are [0, -1, 1] and we take just -1 and 1 (previous and next frame)
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = poses[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.conf['pose_model_type'] == "posecnn" and not self.conf['use_stereo_training']:
                    axisangle = poses[("axisangle", 0, frame_id)]
                    translation = poses[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs_dict[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                         pix_coords,
                                                                         padding_mode="border", #if frame_id == 1 else "border", #zeros? border?
                                                                         align_corners=True)
                if not conf.get('disable_automasking'):
                    outputs_dict[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        return outputs_dict

    def log(self, mode, inputs, outputs_dict, losses, metrics):
        """
            Write an event to the tensorboard events file.
        """

        writer = self.writers[mode]
        for l, v in losses.items():
            if l!="loss/0" and l!="loss/1" and l!="loss/2" and l!="loss/3" and l!="loss/4" and l!="loss/5":
                writer.add_scalar("{}".format(l), v, self.step)

        for m, v in metrics.items():
            writer.add_scalar("{}".format(m), v, self.step)


        for j in range(min(4, self.conf['bs'])):  # write a maxmimum of 4 images
            for frame_id in self.conf['frame_ids_training']:
                if self.step == 0:
                    writer.add_image("input_color_image_{}/{}".format(frame_id, j), inputs[("color", frame_id, 0)][j].data)
                if frame_id!=0 and self.step%100==0: 
                    writer.add_image("warped_color_image_{}/{}".format(frame_id, j), outputs_dict[("color", frame_id, 0)][j].data, self.step)


            if not self.conf.get('disable_automasking'):
              writer.add_image("automask/{}".format(j), outputs_dict["identity_selection/{}".format(0)][j][None, ...], self.step)


            writer.add_image("predicted_disp/{}".format(j), normalize_image(outputs_dict[("disp_unscaled", 0, 0)][j]), self.step)


            # Saving color mapped depth image
            output_depth_map = outputs_dict[("disp_unscaled", 0, 0)][j]
            output_depth_map_np = output_depth_map.squeeze().detach().cpu().numpy()
            vmax = np.percentile(output_depth_map_np, 95) # The 95th percentile is used here to ignore the top 5% of depth values which might be outliers.
                                                          # This is a common practice to enhance the contrast of the visualization by focusing on the range where most of the data lies.
            normalizer = mpl.colors.Normalize(vmin=output_depth_map_np.min(), vmax=vmax) # scale data values to the range [0, 1]. 
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')  # choices: [cmap='viridis', cmap='plasma_r'].
            color_depth_map = (mapper.to_rgba(output_depth_map_np)[:, :, :3] * 255).astype(np.uint8) # (375, 1242, 3)
            writer.add_image("predicted_depth_map_color/{}".format(j), color_depth_map.transpose(2,0,1), self.step)


    def save_opts(self):
        """
            Save options to tensorboard so we know what we ran this experiment with.
        """

        # depth_encoder_decoder = summary(self.models["depth_encoder_decoder"], torch.rand(1, 3, self.conf['im_sz'][0], self.conf['im_sz'][1]).cuda(),
        #             max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['overfit'].add_text('depth encoder_decoder', depth_encoder_decoder.__repr__())

        # x1=torch.rand(1, 3, self.conf['im_sz'][1], self.conf['im_sz'][0]).cuda()
        # x2=torch.rand(1, 3, self.conf['im_sz'][1], self.conf['im_sz'][0]).cuda()
        # input_pose_cnn=torch.cat((x1, x2), dim=1)
        # pose_cnn = summary(self.models["pose_cnn"], input_pose_cnn, max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['overfit'].add_text('pose cnn', pose_cnn.__repr__())

        self.writers['overfit'].add_text('config', self.conf.__repr__())



if __name__ == "__main__":
    lt.monkey_patch()

    conf = Conf().conf  

    overfit = Overfit(conf)
    overfit.overfit_batch()

