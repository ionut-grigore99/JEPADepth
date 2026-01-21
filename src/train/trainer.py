from __future__ import absolute_import, division, print_function

import time
import yaml
import lovely_tensors as lt
from datetime import datetime
import torch.optim as optim
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_model_summary import summary

from src.config.conf import TrainConf
from src.models.posenet.pose_cnn import PoseCNN
from src.models.posenet.pose_cnn_monodepth2 import MonoDepth2PoseCNN, ResnetEncoder, PoseDecoder
from src.datasets.kitti_dataset import KITTIRAWDataset, KITTIOdomDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *

class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.get = lambda x: conf.get(x)

        self.log_path = os.path.join(self.get('tensorboard_path'), 'train')
        self.log_path = os.path.join(self.log_path, self.get('model_name'))
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))   

        # checking height and width are multiples of 32
        assert self.get('im_sz')[0] % 32 == 0, "'height' must be a multiple of 32"
        assert self.get('im_sz')[1] % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        # self.models = Models(conf)
        self.parameters_to_train = []

        self.device = torch.device("cuda:7" if self.get('use_cuda') else "cpu")
        #self.device = self.get('device')

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


        if self.get('use_stereo_training'):
            self.get('frame_ids_training').append("s")

        if conf.get('model_name').startswith("mambaDepthBot"):
            self.models["depth_encoder_decoder"] = get_mambaDepth_bot_model(conf)


        elif conf.get('model_name').startswith("mambaDepthEnc"):
            self.models["depth_encoder_decoder"] = get_mambaDepth_enc_model(conf)

        elif conf.get('model_name').startswith("vmunet"):
            self.models["depth_encoder_decoder"] = VMUNet(
                                                            num_classes=1,
                                                            input_channels=3,
                                                            depths=[2,2,2,2],
                                                            depths_decoder=[2,2,2,2],
                                                            drop_path_rate=0.2,
                                                            load_ckpt_path='/data/disertatie/MambaDepth/src/pretrained/vmamba_small_e238_ema.pth' # vssmbase_dp06_ckpt_epoch_241.pth, vmamba_small_e238_ema.pth
                                                        )
            # self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"]
            self.models["depth_encoder_decoder"].load_from()

        elif conf.get('model_name').startswith("monodepth2"):
            self.models["depth_encoder_decoder"] = MonoDepth2EncoderDecoder()
            # self.models["depth_encoder_decoder"]=self.models["depth_encoder_decoder"].cuda()

        else:
            print("Chose one of these 2 models: mambdaDepthEnc and mambaDepthBot!")
            exit()

        if self.get('load_pretrained_model'):
            self.models["depth_encoder_decoder"].from_pretrained()
 

        if isinstance(self.device, list):
            self.models["depth_encoder_decoder"].to(self.device[0]) # move to first device
            self.models["depth_encoder_decoder"] = torch.nn.DataParallel(self.models["depth_encoder_decoder"], device_ids=self.device)
            # self.models["depth_encoder_decoder"] = torch.nn.DataParallel(self.models["depth_encoder_decoder"].cuda(), device_ids=self.device, output_device=self.device[0])
        else:
            self.models["depth_encoder_decoder"].to(self.device)

        #self.models["depth_encoder_decoder"] = torch.nn.DataParallel(self.models["depth_encoder_decoder"])
        self.parameters_to_train += list(self.models["depth_encoder_decoder"].parameters())

        if conf.get('pose_model_type')=="pose_cnn":
            self.models["pose_cnn"] = PoseCNN(self.num_pose_frames) # default=2
            if self.get('load_pretrained_pose'):
                self.models["pose_cnn"].from_pretrained(weights_path='/data/disertatie/SQLdepth/src/pretrained/KITTI_EfficientNetB5_320x1024_models/pose.pth', device=self.device)

        else:
            self.models["pose_cnn_encoder"] = ResnetEncoder(num_layers=18, pretrained=True, num_input_images=2)
            self.models["pose_cnn_decoder"] = PoseDecoder(num_ch_enc=np.array([64, 64, 128, 256, 512]),  num_input_features=1, num_frames_to_predict_for=2)
            if self.get('load_pretrained_pose'):
                self.models["pose_cnn_encoder"].from_pretrained()
                self.models["pose_cnn_decoder"].from_pretrained()

        if isinstance(self.device, list):
            self.models["pose_cnn_encoder"].to(self.device[0]) # move to first device
            # self.models["pose_cnn_encoder"] = torch.nn.DataParallel(self.models["pose_cnn_encoder"].cuda(), device_ids=self.device, output_device=self.device[0])
        else:
            self.models["pose_cnn_encoder"].to(self.device)
        #self.models["pose_cnn_encoder"] = torch.nn.DataParallel(self.models["pose_cnn_encoder"], device_ids=self.device)
        #self.models["pose_cnn_encoder"] = torch.nn.DataParallel(self.models["pose_cnn_encoder"])
        self.parameters_to_train += list(self.models["pose_cnn_encoder"].parameters())

        if isinstance(self.device, list):
            self.models["pose_cnn_decoder"].to(self.device[0]) # move to first device
            # self.models["pose_cnn_decoder"] = torch.nn.DataParallel(self.models["pose_cnn_decoder"].cuda(), device_ids=self.device, output_device=self.device[0])
        else:
            self.models["pose_cnn_decoder"].to(self.device)
        #self.models["pose_cnn_decoder"] = torch.nn.DataParallel(self.models["pose_cnn_decoder"], device_ids=self.device)
        #self.models["pose_cnn_decoder"] = torch.nn.DataParallel(self.models["pose_cnn_decoder"])
        self.parameters_to_train += list(self.models["pose_cnn_decoder"].parameters())



        # self.models.to(self.device[0]) # move to first device
        # self.models = torch.nn.DataParallel(self.models, device_ids=self.device)
        # self.parameters_to_train += list(self.models.depth_encoder_decoder.parameters())
        # self.parameters_to_train += list(self.models.pose_cnn_emcoder.parameters())
        # self.parameters_to_train += list(self.models.pose_cnn_decoder.parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.get('learning_rate'), weight_decay=self.get('weight_decay')) # default=1e-4

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.get('scheduler_step_size'), 0.1) # default=15


        print("Training model named:\n  ", self.get('model_name'))
        print("Models and tensorboard events files are saved to:\n  ", self.get('tensorboard_path'))
        print("Training is using:\n  ", self.device)
        print("Using split:\n  ", self.get('training_split'))

        # Preparing data for training
        datasets_dict = {"kitti": KITTIRAWDataset, "kitti_odom": KITTIOdomDataset}
        self.dataset = datasets_dict[self.get('training_dataset')] # default="kitti"

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/kitti/kitti_splits", self.get('training_split'), "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.get('train_from_png') else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.get('bs') * self.get('num_epochs') # total number of iterations

        train_dataset = self.dataset(self.get('data_path'), train_filenames, self.get('im_sz')[0], self.get('im_sz')[1],
                                     self.get('frame_ids_training'), self.num_scales, is_train=True, img_ext=img_ext) # 1 means num_scales
        self.train_loader = DataLoader(train_dataset, self.get('bs'), True, num_workers=self.get('num_workers'), pin_memory=True, drop_last=True)

        val_dataset = self.dataset(self.get('data_path'), val_filenames, self.get('im_sz')[0], self.get('im_sz')[1],
                                   self.get('frame_ids_training'), self.num_scales, is_train=False, img_ext=img_ext) # 1 means num_scales
        self.val_loader = DataLoader(val_dataset, self.get('bs'), True, num_workers=self.get('num_workers'), pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images. We always use it.
        if self.get('use_ssim'):
            self.ssim = SSIM()
            if isinstance(self.device, list):
                self.ssim.to(self.device[0])
            else:
                self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.get('loss_scales'): # default we have just 1 scale which is set to 0!
            h = self.get('im_sz')[0] // (2 ** scale) # this will do nothing basically
            w = self.get('im_sz')[1] // (2 ** scale) # also this

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.get('bs'), h, w)
            if isinstance(self.device, list):
                self.backproject_depth[scale].to(self.device[0])
            else:
                self.backproject_depth[scale].to(self.device)


            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.get('bs'), h, w)
            if isinstance(self.device, list):
                self.project_3d[scale].to(self.device[0])
            else:
                self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        self.save_opts()

    def set_train(self):
        """
            Convert all models to training mode.
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """
            Convert all models to testing/evaluation mode.
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
            Run the entire training pipeline.
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.save_model()
        for self.epoch in range(self.get('num_epochs')):
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.get('save_frequency') == 0:
                self.save_model()

    def run_epoch(self):
        """
            Run a single epoch of training and validation.
        """
        print("Training")
        self.set_train()

        metrics = {}

        for batch_idx, inputs in enumerate(self.train_loader):
            before_optimization_time = time.time()

            outputs_dict, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            if self.get('clip_grad_norm'): torch.nn.utils.clip_grad_norm_(self.models["depth_encoder_decoder"].parameters(), 1)
            self.model_optimizer.step()

            duration_optimization = time.time() - before_optimization_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.get('log_frequency') == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0
            # Practic fac log din 10 in 10 batch-uri daca step<2000, daca nu fac la fiecare 1000 steps.
            # 1 step = 1 iteratie

            if early_phase or late_phase:
                self.log_time(batch_idx, duration_optimization, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    compute_depth_metrics(inputs, outputs_dict, metrics, self.depth_metric_names)
                    #basically  it computes depth metrics for a batch (inputs is a batch)
                self.log("train", inputs,  outputs_dict, losses, metrics)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
        """
        for key, ipt in inputs.items():
            if isinstance(self.device, list):
                inputs[key] = ipt.to(self.device[0])
            else:
                inputs[key] = ipt.to(self.device)

        if self.get('pose_model_type') == "shared": # default no
            # If we are using a shared encoder for both depth and pose (as advocated in monodepthv1),
            # then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.get('frame_ids_training')])
            all_features = self.models["depth_encoder_decoder"].encoder(all_color_aug)
            all_features = [torch.split(f, self.get('batch_size')) for f in all_features]

            features = {}
            for i, k in enumerate(self.get('frame_ids_training')):
                features[k] = [f[i] for f in all_features]
            disp_maps = self.models["depth_encoder_decoder"].decoder(features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["depth_encoder_decoder"](inputs["color_aug", 0, 0])
            disp_maps = features

        poses=None
        if self.use_pose_net: # default=True
            poses=predict_poses(conf, self.models, inputs, features)

        outputs_dict = self.generate_images_pred(inputs, disp_maps, poses)
        losses = compute_losses(self.conf, inputs, disp_maps, outputs_dict, self.ssim)

        return outputs_dict, losses


    def val(self):
        """
            Validate the model on a single minibatch.
        """
        self.set_eval()
        
        metrics = {}
        
        try:
            inputs = next(self.val_iter) # for new pytorch
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs_dict, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                compute_depth_metrics(inputs, outputs_dict, metrics, self.depth_metric_names)
            self.log("val", inputs, outputs_dict, losses, metrics)
            del inputs, outputs_dict, losses

        self.set_train()

    def generate_images_pred(self, inputs, disp_maps, poses):
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
        outputs_dict = {}

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
            
            if self.get('monodepthv1_multiscale'):
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.get('im_sz')[0], self.get('im_sz')[1]], mode="bicubic", align_corners=True) #modified here 
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.get('frame_ids_training')[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = poses[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.get('pose_model_type') == "posecnn" and not self.get('use_stereo_training'):

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

                if not self.get('disable_automasking'):
                    outputs_dict[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        return outputs_dict


    def log_time(self, batch_idx, duration, loss):
        """
            Print a logging statement to the terminal.
        """
        samples_per_sec = self.get('bs') / duration
        time_so_far = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_so_far if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, sec_to_hm_str(time_so_far), sec_to_hm_str(training_time_left)))

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


        for j in range(min(4, self.get('bs'))):  # write a maxmimum of 4 images
            for frame_id in self.get('frame_ids_training'):
                if self.step == 0:
                    writer.add_image("input_color_image_{}/{}".format(frame_id, j), inputs[("color", frame_id, 0)][j].data)
                if frame_id!=0 and self.step%500==0: 
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

        # depth_encoder_decoder = summary(self.models["depth_encoder_decoder"], torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]).cuda(),
        #             max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['train'].add_text('depth encoder_decoder', depth_encoder_decoder.__repr__())

        # x1=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]).cuda()
        # x2=torch.rand(1, 3, self.get('im_sz')[1], self.get('im_sz')[0]).cuda()
        # input_pose_cnn=torch.cat((x1, x2), dim=1)
        # pose_cnn = summary(self.models["pose_cnn"], input_pose_cnn, max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['train'].add_text('pose cnn', pose_cnn.__repr__())

        self.writers['train'].add_text('config', self.conf.__repr__())


    def save_model(self):
        """
            Save model weights to disk.
        """
        save_folder = os.path.join(self.log_path, "models", "weights_epoch_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))

            # for nn.DataParallel models, you must use model.module.state_dict() instead of model.state_dict()!
            if model_name == 'pose_cnn_encoder':
               to_save = model.state_dict()
            elif model_name == 'pose_cnn_decoder':
                to_save = model.state_dict()
            else:
                if isinstance(self.device, list): # NOTE: BE VERY CAREFUL TO NOT FORGET THIS!
                    to_save = model.module.state_dict()
                else:
                    to_save = model.state_dict()
            

            if model_name == 'encoder':
                # save the input sizes - these are needed at prediction time
                to_save['height'] = self.get('im_sz')[0]
                to_save['width'] = self.get('im_sz')[1]
                to_save['use_stereo'] = self.get('use_stereo_training')
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


if __name__ == "__main__":

    lt.monkey_patch()

    conf = TrainConf().conf  

    trainer = Trainer(conf)
    trainer.train()

