Încearcă AI direct în aplicațiile preferate … 
Folosește Gemini pentru a genera schițe și a rafina conținut și beneficiază de Gemini Pro cu acces la AI de ultimă generație de la Google la 109,99 RON 13,99 RON pentru 3 luni (preț personalizat)

from __future__ import absolute_import, division, print_function

import lovely_tensors as lt
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_model_summary import summary

from ..config.conf import MambaDepthBot_320x1024_Conf, MambaDepthEnc_320x1024_Conf, OverfitConf
from ..models.mamba.MambaDepthBot import MambaDepthBot, get_mambaDepth_bot_model
from ..models.mamba.MambaDepthEnc import MambaDepthEnc, get_mambaDepth_enc_model
from ..models.monodepth2.MonoDepth2 import MonoDepth2EncoderDecoder
from ..models.posenet.pose_cnn import PoseCNN
from ..models.exp2 import VMUNet
from ..models.exp3 import MambaCustom
from ..models.exp4 import DeepNet
from ..models.exp import Unet
from ..models.posenet.pose_cnn_monodepth2 import MonoDepth2PoseCNN, ResnetEncoder, PoseDecoder
from ..datasets.kitti_dataset import KITTIRAWDataset, KITTIOdomDataset
from ..utils import *
from ..models.layers import *
from ..losses.loss import *
from ..data.kitti.kitti_utils.kitti_utils import *


class Overfit:
    def __init__(self, conf):
        self.conf = conf
        self.get = lambda x: conf.get(x)

        self.log_path = os.path.join(self.get('tensorboard_path'), 'overfit')
        self.log_path = os.path.join(self.log_path, self.get('model_name'))
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))

        # checking height and width are multiples of 32
        assert self.get('im_sz')[0] % 32 == 0, "'height' must be a multiple of 32"
        assert self.get('im_sz')[1] % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if self.get('use_cuda') else "cpu")

        self.num_scales = len(self.get('loss_scales')) # default=[0], we only perform single scale training => num_scales=1
        self.num_input_frames = len(self.get('frame_ids_training')) # default=[0, -1, 1] => num_input_frames=3
        self.num_pose_frames = 2 if self.get('pose_model_input') == "pairs" else self.num_input_frames # default=2

        self.min_depth = self.get('min_depth') #cat sa pun aici sa fie properly?? momentan e 0.1
        self.max_depth = self.get('max_depth') #cat sa pun aici sa fie properly?? momentan e 100.0

        assert self.get('frame_ids_training')[0] == 0, "frame_ids_training must start with 0"

        self.use_pose_net = not (self.get('use_stereo_training') and self.get('frame_ids_training') == [0])
        # the parenthesis will be always False because frame_ids_training will never be [0] and thus
        # self.use_pose_net will be True always. I think self.get('frame_ids_training') == [0] only in supervised
        # settings and thus we basically use PoseNet when we have not stereo training and also we haven't supervised training.


        if self.get('use_stereo_training'): #this will not happen because use_stereo_training is set to False (??)
            self.get('frame_ids_training').append("s")

        if conf.get('model_name').startswith("mambaDepthBot"):
            self.models["depth_encoder_decoder"] = get_mambaDepth_bot_model(conf)
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        elif conf.get('model_name').startswith("mambaDepthEnc"):
            self.models["depth_encoder_decoder"] = get_mambaDepth_enc_model(conf)
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        elif conf.get('model_name').startswith("monodepth2"):
            self.models["depth_encoder_decoder"] = MonoDepth2EncoderDecoder()
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        elif conf.get('model_name').startswith("monovit"):
            self.models["depth_encoder_decoder"]=DeepNet(type = 'mpvitnet', weights_init= "pretrained", num_layers=18, num_pose_frames=2, scales=range(4))
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()
        
        elif conf.get('model_name').startswith("vmunet"):
            self.models["depth_encoder_decoder"] = VMUNet(
                                                            num_classes=1,
                                                            input_channels=3,
                                                            depths=[2,2,2,2],
                                                            depths_decoder=[2,2,2,2],
                                                            drop_path_rate=0.2,
                                                            load_ckpt_path='/data/disertatie/MambaDepth/src/pretrained/vssmbase_dp06_ckpt_epoch_241.pth'  # vssmbase_dp06_ckpt_epoch_241.pth, vmamba_small_e238_ema.pth
                                                        )
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].to('cuda:6')
            self.models["depth_encoder_decoder"].load_from()


        elif conf.get('model_name').startswith("mambaCustom"):
            self.models["depth_encoder_decoder"] = MambaCustom(conf)
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        elif conf.get('model_name').startswith("unet"):
            self.models["depth_encoder_decoder"] = Unet(pretrained=True, backbone="tf_efficientnet_b5_ap", in_channels=3, num_classes=1, decoder_channels=[512, 256, 128, 64, 32])
            self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        else:
            print("Chose one of these 2 models: mambdaDepthEnc and mambaDepthBot!")
            exit()
            

        if self.get('load_pretrained_model'):
            self.models["depth_encoder_decoder"].from_pretrained()

        self.models["depth_encoder_decoder"] = self.models["depth_encoder_decoder"].to('cuda:6')#self.device)
        self.parameters_to_train += list(self.models["depth_encoder_decoder"].parameters())


        if conf.get('pose_model_type')=="pose_cnn":
            self.models["pose_cnn"] = PoseCNN(self.num_pose_frames) # default=2
            if self.get('load_pretrained_pose'):
                self.models["pose_cnn"].from_pretrained(weights_path='/data/disertatie/SQLdepth/src/pretrained/KITTI_EfficientNetB5_320x1024_models/pose.pth', device= 'cuda:6')#self.device)

        else:
            self.models["pose_cnn_encoder"] = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=2)
            self.models["pose_cnn_decoder"] = PoseDecoder(num_ch_enc=np.array([64, 64, 128, 256, 512]),  num_input_features=1, num_frames_to_predict_for=2)
            if self.get('load_pretrained_pose'):
                self.models["pose_cnn_encoder"].from_pretrained(weights_path='/data/disertatie/MambaDepth/src/pretrained/pose_encoder.pth', device='cuda:6')
                self.models["pose_cnn_decoder"].from_pretrained(weights_path='/data/disertatie/MambaDepth/src/pretrained/pose.pth', device='cuda:6')



        self.models["pose_cnn_encoder"] = self.models["pose_cnn_encoder"].to('cuda:6')  #self.device)
        self.models["pose_cnn_decoder"] = self.models["pose_cnn_decoder"].to('cuda:6')  #self.device)
        # self.parameters_to_train += list(self.models["pose_cnn_encoder"].parameters()) # I think that in case of overfit this must be commented!
        # self.parameters_to_train += list(self.models["pose_cnn_decoder"].parameters()) # I think that in case of overfit this must be commented!

            
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.get('learning_rate')) # default=1e-4

        print("Overfiting model named:\n  ", self.get('model_name'))
        print("Models and tensorboard events files are saved to:\n  ", self.get('tensorboard_path'))
        print("Overfiting is using:\n  ", self.device)
        print("Using split:\n  ", self.get('training_split'))

        # Preparing data for training
        datasets_dict = {"kitti": KITTIRAWDataset, "kitti_odom": KITTIOdomDataset}
        self.dataset = datasets_dict[self.get('training_dataset')] # default="kitti"

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/kitti/kitti_splits", self.get('training_split'), "{}_files.txt")

        overfit_filenames = readlines(fpath.format("overfit"))
        img_ext = '.png' if self.get('train_from_png') else '.jpg'

        overfit_dataset = self.dataset(self.get('data_path'), overfit_filenames, self.get('im_sz')[0], self.get('im_sz')[1],
                                     self.get('frame_ids_training'), self.num_scales, is_train=True, img_ext=img_ext) # 1 means num_scales
        self.overfit_loader = DataLoader(overfit_dataset, self.get('bs'), False, num_workers=self.get('num_workers'),
                                       pin_memory=True, drop_last=True)

        self.batch = next(iter(self.overfit_loader))

        self.writers = {}
        mode='overfit'
        self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images. We always use it.
        if self.get('use_ssim'):
            self.ssim = SSIM()
            self.ssim.to('cuda:6')  #self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.get('loss_scales'): # default we have just 1 scale which is set to 0!
            h = self.get('im_sz')[0] // (2 ** scale) # this will do nothing basically
            w = self.get('im_sz')[1] // (2 ** scale) # also this

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.get('bs'), h, w)
            self.backproject_depth[scale].to('cuda:6')  #self.device)

            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.get('bs'), h, w)
            self.project_3d[scale].to('cuda:6') #self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        self.save_opts()

    def overfit_batch(self, iters=2000):
        self.step=0
        metrics = {}
        for i in (tbar := tqdm(range(iters), desc="Overfit")):
            self.model_optimizer.zero_grad()
            outputs_dict, losses = self.process_batch(self.batch) # aici e important ce se intampla!
            if "depth_gt" in self.batch:
                compute_depth_metrics(self.batch, outputs_dict, metrics, self.depth_metric_names) 
            #  Write to the tensorboard events file.
            self.log("overfit", self.batch, outputs_dict, losses, metrics)
            losses["loss"].backward()
            if self.get('clip_grad_norm'): torch.nn.utils.clip_grad_norm_(self.models["depth_encoder_decoder"].parameters(), 1)
            self.model_optimizer.step()
            self.step += 1
        print("Overfit done!")

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to('cuda:6')  #self.device)

        if self.get('pose_model_type') == "shared": # default no, our pose_model_type is posecnn
            # If we are using a shared encoder for both depth and pose (as advocated in Monodepthv1),
            # then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.get('frame_ids_training')])
            all_features = self.models["depth_encoder_decoder"].encoder(all_color_aug)
            all_features = [torch.split(f, self.get('batch_size')) for f in all_features]

            features = {}
            for i, k in enumerate(self.get('frame_ids_training')):
                features[k] = [f[i] for f in all_features]
            disp_maps = self.models["depth_encoder_decoder"].depth_decoder(features[0])
        else: # we always go through this branch!
            # Otherwise, we only feed the augmented image with frame_id 0 through the depth encoder
            if conf.get('model_name').startswith("mambaDepth"):
                features = self.models["depth_encoder_decoder"](inputs["color_aug", 0, 0]) # ["color_aug", 0, 0] means frame_id 0 and scale 0 (which is full resolution 320x1024)
                disp_maps = features # the output of our depth_encoder_decoder is disparity which is 1/depth

            elif conf.get('model_name').startswith("monodepth"):# for Monodepth2
                features = self.models["depth_encoder_decoder"].encoder(inputs["color_aug", 0, 0]) # ["color_aug", 0, 0] means frame_id 0 and scale 0 (which is full resolution 320x1024)
                disp_maps = self.models["depth_encoder_decoder"].decoder(features) # the output of our depth_encoder_decoder is disparity which is 1/depth
            
            else:
                features = self.models["depth_encoder_decoder"](inputs["color_aug", 0, 0]) # ["color_aug", 0, 0] means frame_id 0 and scale 0 (which is full resolution 320x1024)
                disp_maps = features # the output of our depth_encoder_decoder is disparity which is 1/depth

        #breakpoint()

        poses=None
        # Predict poses between input frames for monocular sequences.
        # Basically in our configurations we don't use features in predict_poses function, because
        # in our case pose_model_type is "pose_cnn" and num_pose_frames=2.
        if self.use_pose_net: # default=True
            poses=predict_poses(conf, self.models, inputs, features) # We don't use features in this function! We predicts poses
                                                                     # just by giving the concatenated frames as input to PoseNet.

        outputs_dict = self.generate_warped_images(inputs, disp_maps, poses) # See the diagram of self-supervised MDE pipeline. 
        # 'outputs_dict' dictionary contains the warped (reprojected) color images and the predicted depth_maps interpolated at the 
        #  original resolution (because the net predicts at halved resolution) and the color input frame.

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

        for scale in self.get('loss_scales'):
            
            
            if conf.get('model_name').startswith("mambaDepth"):
                disp = disp_maps[scale]
            elif conf.get('model_name').startswith("monodepth"):
                disp = disp_maps[("disp", scale)] # for monodepth2
            elif conf.get('model_name').startswith("mambaCustom"):
                disp = disp_maps[scale]
            elif conf.get('model_name').startswith("monovit"):
                disp = disp_maps[("disp", scale)]
            else:
                disp = disp_maps[scale]

            #breakpoint()
            if self.get('monodepthv1_multiscale'):
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.get('im_sz')[0], self.get('im_sz')[1]], mode="bicubic", align_corners=True)
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            #breakpoint()

            for i, frame_id in enumerate(self.get('frame_ids_training')[1:]): # basically frame_ids_training are [0, -1, 1] and we take just -1 and 1 (previous and next frame)
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = poses[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.get('pose_model_type') == "posecnn" and not self.get('use_stereo_training'):
                    assert False

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
                #print(v)

        for m, v in metrics.items():
            writer.add_scalar("{}".format(m), v, self.step)


        for j in range(min(4, self.get('bs'))):  # write a maxmimum of 4 images
            for frame_id in self.get('frame_ids_training'):
                if self.step == 0:
                    writer.add_image("input_color_image_{}/{}".format(frame_id, j), inputs[("color", frame_id, 0)][j].data)
                if frame_id!=0 and self.step%100==0: 
                    writer.add_image("warped_color_image_{}/{}".format(frame_id, j), outputs_dict[("color", frame_id, 0)][j].data, self.step)


            if not self.get('disable_automasking'):
              writer.add_image("automask/{}".format(j), outputs_dict["identity_selection/{}".format(0)][j][None, ...], self.step)


            writer.add_image("predicted_disp/{}".format(j), normalize_image(outputs_dict[("disp_unscaled", 0, 0)][j]), self.step)


            # Saving color mapped depth image
            output_depth_map = outputs_dict[("disp_unscaled", 0, 0)][j]
            output_depth_map_np = output_depth_map.squeeze().detach().cpu().numpy()
            vmax = np.percentile(output_depth_map_np, 95)
            # The 95th percentile is used here to ignore the top 5% of depth values which might be outliers.
            # This is a common practice to enhance the contrast of the visualization by focusing on the range where most of the data lies.
            normalizer = mpl.colors.Normalize(vmin=output_depth_map_np.min(), vmax=vmax) # scale data values to the range [0, 1]. I
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')  # choices: [cmap='viridis', cmap='plasma_r'].
            color_depth_map = (mapper.to_rgba(output_depth_map_np)[:, :, :3] * 255).astype(np.uint8) # (375, 1242, 3)
            writer.add_image("predicted_depth_map_color/{}".format(j), color_depth_map.transpose(2,0,1), self.step)


    def save_opts(self):
        """
            Save options to tensorboard so we know what we ran this experiment with.
        """

        # depth_encoder_decoder = summary(self.models["depth_encoder_decoder"], torch.rand(1, 3, self.get('im_sz')[0], self.get('im_sz')[1]).cuda(),
        #             max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['overfit'].add_text('depth encoder_decoder', depth_encoder_decoder.__repr__())

        # x1=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]).cuda()
        # x2=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]).cuda()
        # input_pose_cnn=torch.cat((x1, x2), dim=1)
        # pose_cnn = summary(self.models["pose_cnn"], input_pose_cnn, max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['overfit'].add_text('pose cnn', pose_cnn.__repr__())

        self.writers['overfit'].add_text('config', self.conf.__repr__())



if __name__ == "__main__":

    lt.monkey_patch()

    conf = OverfitConf().conf  # MambaDepthBot_320x1024_Conf, MambaDepthEnc_320x1024_Conf

    overfit = Overfit(conf)
    overfit.overfit_batch()

