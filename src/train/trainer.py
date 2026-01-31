from __future__ import absolute_import, division, print_function

from pathlib import Path
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

from src.config.conf import Conf
from src.models.posenet.simple_pose_cnn import SimplePoseCNN
from src.models.posenet.resnet_pose_cnn import ResNetPoseCNN 
from src.models.pixio.dpt import DPTDepth
from src.models.layers import *
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *

class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.log_path = os.path.join(self.conf['tensorboard_path'], 'train')
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
            self.models["depth_model"] = DPTDepth(self.conf['pixio']['encoder'], self.conf['pixio']['pretrained_ckp'], scales=self.conf['pixio']['scales'])
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

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.conf['learning_rate'], weight_decay=self.conf['weight_decay']) 
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.conf['scheduler_step_size'], 0.1) 

        print("Training model named:\n  ", self.conf['model_name'])
        print("Models and tensorboard events files are saved to:\n  ", self.conf['tensorboard_path'])
        print("Training is using:\n  ", self.device)
        print("Using split:\n  ", self.conf['training_split'])

        # Preparing data for training
        self.dataset = KITTIRAWDataset

        data_path = Path(self.conf["data_path"])
        train_filenames = readlines(data_path.parent / "kitti_splits" / self.conf['training_split'] / "train_files.txt")
        val_filenames = readlines(data_path.parent / "kitti_splits" / self.conf['training_split'] / "val_files.txt")
        img_ext = '.png' if self.conf['train_from_png'] else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.conf['bs'] * self.conf['num_epochs'] # total number of iterations

        train_dataset = self.dataset(self.conf['data_path'], train_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=True, img_ext=img_ext) 
        self.train_loader = DataLoader(train_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)

        val_dataset = self.dataset(self.conf['data_path'], val_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=False, img_ext=img_ext) 
        self.val_loader = DataLoader(val_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images. We always use it.
        if self.conf['use_ssim']:
            self.ssim = SSIM()
            if isinstance(self.device, list):
                self.ssim.to(self.device[0])
            else:
                self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.conf['loss_scales']: # default we have just 1 scale which is set to 0!
            h = self.conf['im_sz'][0] // (2 ** scale) # this will do nothing basically
            w = self.conf['im_sz'][1] // (2 ** scale) # also this

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.conf['bs'], h, w)
            if isinstance(self.device, list):
                self.backproject_depth[scale].to(self.device[0])
            else:
                self.backproject_depth[scale].to(self.device)


            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.conf['bs'], h, w)
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
        for self.epoch in range(self.conf['num_epochs']):
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.conf['save_frequency'] == 0:
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
            if self.conf['clip_grad_norm']: torch.nn.utils.clip_grad_norm_(self.models["depth_encoder_decoder"].parameters(), 1)
            self.model_optimizer.step()

            duration_optimization = time.time() - before_optimization_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.conf['log_frequency'] == 0 and self.step < 2000
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

        # We feed only the image with frame_id 0 through the depth model
        disp_maps = self.models["depth_model"](inputs["color_aug", 0, 0])

        # Predict poses between input frames for monocular sequences.
        poses=predict_poses(conf, self.models, inputs)

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
            inputs = next(self.val_iter) # for new PyTorch
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
        for scale in self.conf['loss_scales']:

            disp = disp_maps[("disp", scale)]
            
            if self.conf['monodepthv1_multiscale']:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.conf['im_sz'][0], self.conf['im_sz'][1]], mode="bicubic", align_corners=True) #modified here
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.conf['frame_ids_training'][1:]):

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

                if not self.conf['disable_automasking']:
                    outputs_dict[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

        return outputs_dict


    def log_time(self, batch_idx, duration, loss):
        """
            Print a logging statement to the terminal.
        """
        samples_per_sec = self.conf['bs'] / duration
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


        for j in range(min(4, self.conf['bs'])):  # write a maxmimum of 4 images
            for frame_id in self.conf['frame_ids_training']:
                if self.step == 0:
                    writer.add_image("input_color_image_{}/{}".format(frame_id, j), inputs[("color", frame_id, 0)][j].data)
                if frame_id!=0 and self.step%500==0: 
                    writer.add_image("warped_color_image_{}/{}".format(frame_id, j), outputs_dict[("color", frame_id, 0)][j].data, self.step)

            if not self.conf['disable_automasking']:
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

        # depth_encoder_decoder = summary(self.models["depth_encoder_decoder"], torch.rand(1, 3, self.conf['im_sz'][1], self.conf['im_sz'][0]).cuda(),
        #             max_depth=4, show_parent_layers=True, print_summary=True)
        # self.writers['train'].add_text('depth encoder_decoder', depth_encoder_decoder.__repr__())

        # x1=torch.rand(1, 3, self.conf['im_sz'][1], self.conf['im_sz'][0]).cuda()
        # x2=torch.rand(1, 3, self.conf['im_sz'][1], self.conf['im_sz'][0]).cuda()
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
                to_save['height'] = self.conf['im_sz'][0]
                to_save['width'] = self.conf['im_sz'][1]
                to_save['use_stereo'] = self.conf['use_stereo_training']
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)


if __name__ == "__main__":
    lt.monkey_patch()

    conf = Conf().conf  

    trainer = Trainer(conf)
    trainer.train()

