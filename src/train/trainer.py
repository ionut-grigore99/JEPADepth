from csv import writer
from pathlib import Path
import time
import yaml
import lovely_tensors as lt
from datetime import datetime
import copy
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import textwrap
from ruamel.yaml import YAML
from io import StringIO

from src.config.conf import Conf
from src.models.posenet.simple_pose_cnn import SimplePoseCNN
from src.models.posenet.resnet_pose_cnn import ResNetPoseCNN 
from src.models.pixio.dpt import DPTDepth
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.models.layers import *
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *

# JEPA imports
from src.masks.mask_collator import MaskCollator as MBMaskCollator
from src.utils import apply_masks, repeat_interleave_batch, trunc_normal_
import src.models.ijepa.vision_transformer as vit
from src.models.dinov3.hub import dinov3_vits16, dinov3_vitb16, dinov3_vitl16
from src.models.dinov3.fpn_decoder import FPNDecoder

class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.log_path = os.path.join(self.conf['tensorboard_path'], 'train')
        self.log_path = os.path.join(self.log_path, self.conf['model_name'] if not self.conf['use_jepa_training'] else f"jepa_{self.conf['jepa']['encoder_size']}")
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_scales = len(self.conf['loss_scales']) 
        self.num_input_frames = len(self.conf['frame_ids_training']) # default=[0, -1, 1] => num_input_frames=3
        self.num_pose_frames = 2 if self.conf['pose_model_input'] == "pairs" else self.num_input_frames # default=2
        self.min_depth = self.conf['min_depth']
        self.max_depth = self.conf['max_depth']
        
        # Check if JEPA training is enabled
        self.use_jepa = self.conf['use_jepa_training']

        # Note: JEPA initialization is deferred until after num_total_steps is calculated. Standard model initialization happens here for non-JEPA mode
        if not self.use_jepa:
            print("Standard training mode (No JEPA)")
            if self.conf['model_name'].startswith("pixio"):
                self.models["depth_model"] = DPTDepth(self.conf['pixio']['encoder'], self.conf['pixio']['pretrained_ckp'], scales=self.conf['pixio']['scales'])
                self.models["depth_model"].from_pretrained(weights_path=self.conf['pixio']['weights_path'], device=self.device) if self.conf['load_pretrained_depth_model'] else None
            elif self.conf['model_name'] == "monodepth2":
                self.models["depth_model"] = MonoDepth2(num_layers=self.conf['monodepth2']['num_layers'], pretrained=self.conf['monodepth2']['pretrained'], scales=self.conf['monodepth2']['scales'])
                self.models["depth_model"].from_pretrained(encoder_weights_path=self.conf['monodepth2']['encoder_weights_path'], decoder_weights_path=self.conf['monodepth2']['decoder_weights_path'], device=self.device) if self.conf['load_pretrained_depth_model'] else None
            else:
                print("Model not recognized!")
                exit()
            self.models["depth_model"] = self.models["depth_model"].to(self.device)
            self.parameters_to_train += list(self.models["depth_model"].parameters())

        # Prepare pose model
        if self.conf['pose_model_type']=="simple_pose_cnn":
            self.models["pose_model"] = SimplePoseCNN(self.num_pose_frames)
            self.models["pose_model"].from_pretrained(weights_path=self.conf['simple_pose_cnn']['weights_path'], device=self.device) if self.conf['load_pretrained_pose_model'] else None
        elif self.conf['pose_model_type']=="resnet_pose_cnn":
            self.models["pose_model"] = ResNetPoseCNN(self.conf['resnet_pose_cnn']['num_layers'], self.conf['resnet_pose_cnn']['pretrained'], self.conf['resnet_pose_cnn']['num_input_images'], self.conf['resnet_pose_cnn']['num_input_features'], self.conf['resnet_pose_cnn']['num_frames_to_predict_for'])
            self.models["pose_model"].from_pretrained(weights_path=self.conf['resnet_pose_cnn']['weights_path'], device=self.device) if self.conf['load_pretrained_pose_model'] else None
        else:
            print("Pose model type not recognized!")
            exit()
        self.models["pose_model"] = self.models["pose_model"].to(self.device)
        self.parameters_to_train += list(self.models["pose_model"].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.conf['learning_rate'], weight_decay=self.conf['weight_decay']) 
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.conf['scheduler_step_size'], self.conf['scheduler_gamma']) 

        print("Training model named:\n  ", self.conf['model_name']) if not self.use_jepa else print("Training JEPA + Depth model with encoder:\n  ", self.conf['jepa']['encoder_size'])
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

        # Now initialize JEPA model if enabled (needs num_total_steps for EMA scheduler)
        if self.use_jepa:
            self.init_jepa_depth_model()

        train_dataset = self.dataset(self.conf['data_path'], train_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=True, img_ext=img_ext) 
        self.train_loader = DataLoader(train_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)

        val_dataset = self.dataset(self.conf['data_path'], val_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=False, img_ext=img_ext) 
        self.val_loader = DataLoader(val_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images.
        if self.conf['use_ssim']:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.conf['loss_scales']:
            h = self.conf['im_sz'][0] // (2 ** scale) 
            w = self.conf['im_sz'][1] // (2 ** scale) 

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.conf['bs'], h, w)
            self.backproject_depth[scale].to(self.device)

            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.conf['bs'], h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        self.save_opts()
    
    def init_jepa_depth_model(self):
        """
        Initialize JEPA architecture with depth decoder.
        
        Components:
        1. Context Encoder (pretrained DINOv3 ViT) - trainable
        2. Target Encoder (EMA copy of context encoder) - frozen
        3. Predictor (small ViT) - trainable
        4. Depth Decoder (FPN-style head) - trainable, attached to context encoder
        """
        jepa_conf = self.conf['jepa']
        
        print("\nInitializing JEPA + Depth Architecture:")
        print("-" * 80)
        
        # Initialize Context Encoder (ViT)
        print("Creating Context Encoder (DINOv3 ViT)...")
        encoder_size = jepa_conf['encoder_size']  
        model_map = {"small": dinov3_vits16, "base": dinov3_vitb16, "large": dinov3_vitl16}

        context_encoder = model_map[encoder_size](pretrained=False)
        weights_path = jepa_conf['encoder_weights_path']
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        context_encoder.load_state_dict(state_dict)
        
        context_encoder.to(self.device)
        self.models["context_encoder"] = context_encoder
        self.parameters_to_train += list(context_encoder.parameters())
        print(f"   Context Encoder: DINOv3 {encoder_size}, embed_dim={context_encoder.embed_dim}, patch_size={context_encoder.patch_size}")
        
        # Initialize Target Encoder (frozen EMA copy)
        print("Creating Target Encoder (frozen EMA copy)...")
        target_encoder = copy.deepcopy(context_encoder)
        target_encoder.to(self.device)
        for param in target_encoder.parameters():
            param.requires_grad = False
        self.models["target_encoder"] = target_encoder
        print(f"   Target Encoder: Frozen copy, parameters not trainable")
        
        # Initialize Predictor
        print("Creating Predictor (lightweight ViT)...")
        predictor_embed_dim = jepa_conf['predictor_emb_dim']
        predictor_depth = jepa_conf['predictor_depth']

        predictor = vit.__dict__['vit_predictor'](
            num_patches=self.conf['im_sz'][0] // context_encoder.patch_size * self.conf['im_sz'][1] // context_encoder.patch_size,
            img_size=self.conf['im_sz'], 
            patch_size=context_encoder.patch_size,
            embed_dim=context_encoder.embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=context_encoder.num_heads,
        )

        # Initialize weights
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
        
        for m in predictor.modules():
            init_weights(m)
        
        predictor.to(self.device)
        self.models["predictor"] = predictor
        self.parameters_to_train += list(predictor.parameters())
        print(f"   Predictor: depth={predictor_depth}, embed_dim={predictor_embed_dim}, predictor_embed_dim={predictor_embed_dim}, num_heads={context_encoder.num_heads}")
        
        # Initialize Depth Decoder
        print("Creating Depth Decoder...")
        depth_decoder = FPNDecoder(
            in_channels=context_encoder.embed_dim,
            decoder_channels=jepa_conf['depth_decoder_channels'],
            scales=self.conf['loss_scales']
        )
        depth_decoder.to(self.device)
        self.models["depth_decoder"] = depth_decoder
        self.parameters_to_train += list(depth_decoder.parameters())
        print(f"   Depth Decoder: decoder_channels={jepa_conf['depth_decoder_channels']}, scales={self.conf['loss_scales']}")
       
        # Mask Collator
        print("Creating Mask Collator...")
        self.mask_collator = MBMaskCollator(
            input_size=(self.conf['im_sz'][0], self.conf['im_sz'][1]),
            patch_size=jepa_conf['patch_size'],
            enc_mask_scale=jepa_conf['enc_mask_scale'],
            pred_mask_scale=jepa_conf['pred_mask_scale'],
            aspect_ratio=jepa_conf['aspect_ratio'],
            nenc=jepa_conf['num_enc_masks'],
            npred=jepa_conf['num_pred_masks'],
            allow_overlap=jepa_conf['allow_overlap'],
            min_keep=jepa_conf['min_keep']
        )
        print(f"   Mask Collator: context {jepa_conf['enc_mask_scale']}, target {jepa_conf['pred_mask_scale']}")
        
        # EMA Momentum Scheduler
        print("Creating EMA Momentum Scheduler...")
        ema_start, ema_end = jepa_conf['ema']
        self.ema_scheduler = iter([
            ema_start + i * (ema_end - ema_start) / self.num_total_steps
            for i in range(self.num_total_steps + 1)
        ])
        print(f"   EMA Scheduler: {ema_start} -> {ema_end} over {self.num_total_steps} steps")

        # JEPA loss weight and meters
        self.jepa_loss_weight = jepa_conf['loss_weight']
        print(f"   JEPA Loss Weight: {self.jepa_loss_weight}")
        
        print("-" * 80)
        print("JEPA + Depth Architecture Initialized Successfully!\n")

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
            
            if self.conf['clip_grad_norm'] is not None: 
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, self.conf['clip_grad_norm'])
            self.model_optimizer.step()

            # Update target encoder with EMA after optimizer step in JEPA mode
            if self.use_jepa:
                current_ema = self.update_target_encoder()
                self.writers["train"].add_scalar("ema_momentum", current_ema, self.step)

            duration_optimization = time.time() - before_optimization_time

            current_lr = self.model_optimizer.param_groups[0]['lr']
            self.writers["train"].add_scalar("learning_rate", current_lr, self.step)

            # log less frequently after the first 2000 steps to save time & disk space:
            #  - log every 10 batches if step < 2000, otherwise log every 1000 steps
            #  - 1 step = 1 iteration = 1 batch
            early_phase = batch_idx % self.conf['log_frequency'] == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration_optimization, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    compute_depth_metrics(inputs, outputs_dict, metrics, self.depth_metric_names) # compute depth metrics for a batch
                self.log("train", inputs,  outputs_dict, losses, metrics)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
            
            If JEPA training:
                - Context Encoder (DINOv3 ViT) -> Depth Decoder -> Disparity Maps
                - Context Encoder -> Predictor -> JEPA Loss (with Target Encoder)
            Else:
                - Depth Model -> Disparity Maps
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.use_jepa: # JEPA mode: Use context encoder + depth decoder            
            # Get images and prepare batch for mask collator
            imgs = inputs["color_aug", 0, 0]
            
            # MaskCollator expects a list of tensors, not a batched tensor. Convert batch to list for mask generation
            batch_list = [imgs[i] for i in range(imgs.shape[0])]
            
            # Generate masks using collator (returns collated_batch, masks_enc, masks_pred)
            _, masks_enc, masks_pred = self.mask_collator(batch_list)
            
            # Move masks to device (they are lists of tensors)
            masks_enc = [u.to(self.device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(self.device, non_blocking=True) for u in masks_pred]

            # Forward through context encoder (DINOv3 ViT). For depth prediction use full features WITHOUT masks
            features = self.models["context_encoder"].get_intermediate_layers(imgs, n=self.num_scales, reshape=True)
            
            # Forward through depth decoder to get disparity maps 
            disp_maps = self.models["depth_decoder"](features)
            
            # Compute JEPA loss (this uses masked encoder outputs)
            jepa_loss = self.compute_jepa_loss(imgs, masks_enc, masks_pred)
            
            # Store masks for visualization
            masks_enc_vis = masks_enc[0] 
            masks_pred_vis = masks_pred[0] 
            
        else: # Standard mode: Use depth model
            disp_maps = self.models["depth_model"](inputs["color_aug", 0, 0])
            jepa_loss = None
            masks_enc_vis = None
            masks_pred_vis = None
           
        # Predict poses (same for both modes)
        poses = predict_poses(self.conf, self.models, inputs)
    
        # Generate warped images and compute photometric loss (same for both modes)
        outputs_dict = self.generate_images_pred(inputs, disp_maps, poses)
        losses = compute_losses(self.conf, inputs, disp_maps, outputs_dict, self.ssim)

        # Add JEPA loss if in JEPA mode
        if self.use_jepa and jepa_loss is not None:
            losses["jepa_loss"] = jepa_loss
            losses["loss"] = losses["loss"] + self.jepa_loss_weight * jepa_loss
            
            # Store masks for visualization
            if masks_enc_vis is not None:
                outputs_dict["jepa_masks_enc"] = masks_enc_vis
            if masks_pred_vis is not None:
                outputs_dict["jepa_masks_pred"] = masks_pred_vis

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
            Apart from these, we also save in the 'outputs_dict' predicted disp_maps (scaled and unscaled) and also depth_maps (1/disp_maps) interpolated at the original resolution.
            'outputs_dict':
                -the predicted disp_maps unscaled interpolated at the original resolution:  outputs_dict[("disp_unscaled", 0, scale)]
                -the predicted disp_maps scaled interpolated at the original resolution:  outputs_dict[("disp_scaled", 0, scale)]
                -the predicted depth_maps interpolated at the original resolution:  outputs_dict[("depth", 0, scale)]
                -the warped (reprojected) color images for a minibatch: outputs_dict[("color", frame_id, scale)] where frame_id is in [-1, 1]
                -optional: the identity warped images (for automasking): outputs_dict[("color_identity", frame_id, scale)] where frame_id is in [-1, 1]
        """
        outputs_dict = {}
        for scale in self.conf['loss_scales']:

            disp = disp_maps[("disp", scale)]
            
            if self.conf['monodepthv1_multiscale']:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.conf['im_sz'][0], self.conf['im_sz'][1]], mode="bicubic", align_corners=True) # changed from "bilinear" to "bicubic" for better quality when upsampling disparity maps to original resolution
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.conf['frame_ids_training'][1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"] # use stereo baseline information
                else:
                    T = poses[("cam_T_cam", 0, frame_id)] # use predicted pose between current frame (0) and neighbor frame
                
                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.conf['pose_model_type'] == "simple_pose_cnn" and not self.conf['use_stereo_training']:

                    axisangle = poses[("axisangle", 0, frame_id)]
                    translation = poses[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs_dict[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                         pix_coords,
                                                                         padding_mode="border",
                                                                         align_corners=True) # align_corners=True for better quality when sampling warped color images

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

    def visualize_masked_image(self, img, mask_indices):
        """
        Visualize image with masked patches grayed out.
        
        Args:
            img: Input image [C, H, W]
            mask_indices: Tensor of patch indices to keep [num_patches_to_keep]
            
        Returns:
            Visualization with masked regions grayed out [C, H, W]
        """
        if mask_indices is None:
            return img
        
        C, H, W = img.shape
        patch_size = self.conf['jepa']['patch_size']
        
        # Calculate number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Create a mask for patches to keep (1 = keep, 0 = mask)
        patch_mask = torch.zeros(num_patches_h * num_patches_w, device=img.device)
        patch_mask[mask_indices] = 1.0
        
        # Reshape to spatial grid
        patch_mask = patch_mask.reshape(num_patches_h, num_patches_w)
        
        # Upsample patch mask to image size
        patch_mask = patch_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches_h, num_patches_w]
        patch_mask = F.interpolate(patch_mask, size=(H, W), mode='nearest')
        patch_mask = patch_mask.squeeze(0)  # [1, H, W]
        
        # Apply mask: keep original where mask=1, gray (0.5) where mask=0
        masked_img = img * patch_mask + 0.5 * (1 - patch_mask)
        
        return masked_img

    def log(self, mode, inputs, outputs_dict, losses, metrics):
        """
            Write an event to the tensorboard events file.
        """
        writer = self.writers[mode]

        for l, v in losses.items():
            if l not in {f"loss/{i}" for i in self.conf['loss_scales']}:
                writer.add_scalar("{}".format(l), v, self.step)
       
        for m, v in metrics.items():
            writer.add_scalar("{}".format(m), v, self.step)


        for j in range(min(4, self.conf['bs'])):  # write a maxmimum of 4 images
            for frame_id in self.conf['frame_ids_training']:
                writer.add_image("input_color_image_{}/{}".format(frame_id, j), inputs[("color", frame_id, 0)][j].data, self.step)
                if frame_id != 0:
                    writer.add_image("warped_color_image_{}/{}".format(frame_id, j), outputs_dict[("color", frame_id, 0)][j].data, self.step)

            if not self.conf['disable_automasking']: # auto-masking stationary pixels visualization
              writer.add_image("automask/{}".format(j), outputs_dict["identity_selection/{}".format(0)][j][None, ...], self.step)

            writer.add_image("predicted_disp/{}".format(j), normalize_image(outputs_dict[("disp_unscaled", 0, 0)][j]), self.step)

            # Saving color mapped depth image
            output_depth_map = outputs_dict[("disp_unscaled", 0, 0)][j]
            output_depth_map_np = output_depth_map.squeeze().detach().cpu().numpy()
            vmax = np.percentile(output_depth_map_np, 95) # The 95th percentile is used here to ignore the top 5% of depth values which might be outliers and to enhance the depth visualization.
            normalizer = mpl.colors.Normalize(vmin=output_depth_map_np.min(), vmax=vmax) 
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')  # choices: ['viridis', 'plasma', 'inferno'].
            color_depth_map = (mapper.to_rgba(output_depth_map_np)[:, :, :3] * 255).astype(np.uint8)
            writer.add_image("predicted_depth_map_color/{}".format(j), color_depth_map.transpose(2,0,1), self.step)
            
            # Log JEPA masked visualizations if in JEPA mode
            if self.use_jepa and "jepa_masks_enc" in outputs_dict and "jepa_masks_pred" in outputs_dict:
                # Get the input image
                input_img = inputs[("color", 0, 0)][j]

                # Visualize context (encoder) masked image
                context_masked = self.visualize_masked_image(input_img, outputs_dict["jepa_masks_enc"][j])
                writer.add_image("jepa_context_masked/{}".format(j), context_masked, self.step)
                
                # Visualize target (predictor) masked image
                target_masked = self.visualize_masked_image(input_img, outputs_dict["jepa_masks_pred"][j])
                writer.add_image("jepa_target_masked/{}".format(j), target_masked, self.step)

    def save_opts(self):
        """
            Save configuration options to tensorboard together with models statistics.
        """
        # Save configuration to TensorBoard exactly as in the file
        yaml = YAML()
        yaml.preserve_quotes = True
        with open("src/config/config.yaml") as f:
            config = yaml.load(f)
        buffer = StringIO()
        yaml.dump(config, buffer)
        # self.writers['train'].add_text('config', buffer.getvalue())
        markdown = f"### Config\n```\n{buffer.getvalue()}\n```" 
        self.writers['train'].add_text('config', markdown)

        
        # Depth Model Statistics
        if self.use_jepa:
            # JEPA mode: separate context encoder + depth decoder
            context_encoder_params = count_parameters(self.models["context_encoder"])
            predictor_params = count_parameters(self.models["predictor"])
            depth_decoder_params = count_parameters(self.models["depth_decoder"])
            target_encoder_params = count_parameters(self.models["target_encoder"])
            
            depth_total_params = (context_encoder_params[0] + predictor_params[0] + 
                                 depth_decoder_params[0] + target_encoder_params[0])
            depth_trainable_params = (context_encoder_params[1] + predictor_params[1] + 
                                     depth_decoder_params[1])  # target_encoder is frozen
            
            # Dynamically determine output shapes by forward pass with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
                dummy_output = self.models["context_encoder"].get_intermediate_layers(dummy_input, n=self.num_scales, reshape=True)
                dummy_output = self.models["depth_decoder"](dummy_output)
            
            depth_stats = textwrap.dedent(f"""
                JEPA + Depth Model Statistics:
                -------------------------------
                Context Encoder (DINOv3 ViT):
                  Total Parameters: {context_encoder_params[0]:,}
                  Trainable Parameters: {context_encoder_params[1]:,}
                
                Target Encoder (EMA):
                  Total Parameters: {target_encoder_params[0]:,}
                  Trainable Parameters: {target_encoder_params[1]:,} (frozen)
                
                Predictor (lightweight ViT):
                  Total Parameters: {predictor_params[0]:,}
                  Trainable Parameters: {predictor_params[1]:,}
                
                Depth Decoder:
                  Total Parameters: {depth_decoder_params[0]:,}
                  Trainable Parameters: {depth_decoder_params[1]:,}
                
                Total (All Components):
                  Total Parameters: {depth_total_params:,}
                  Trainable Parameters: {depth_trainable_params:,}
                  Frozen Parameters: {depth_total_params - depth_trainable_params:,}

                Input Shape: [batch_size, 3, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
                """)
            # Determine if single-scale or multi-scale based on output type
            is_multi_scale = isinstance(dummy_output, dict)
      
            if is_multi_scale:
                depth_stats += "\nOutput Shapes (Multi-Scale):\n"
                for key in sorted(dummy_output.keys()):
                    shape = dummy_output[key].shape
                    depth_stats += f"  {key}: [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
            else:
                shape = dummy_output.shape
                depth_stats += f"Output Shape (Single-Scale): [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
            self.writers['train'].add_text('jepa_depth_model_stats', depth_stats)
        else:
            # Standard mode: single depth model
            depth_total_params, depth_trainable_params = count_parameters(self.models["depth_model"])
            
            # Dynamically determine output shapes by forward pass with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
                dummy_output = self.models["depth_model"](dummy_input)
            
            depth_stats = textwrap.dedent(f"""
                Depth Model Statistics:
                -----------------------
                Model Type: {self.conf['model_name']}
                Total Parameters: {depth_total_params:,}
                Trainable Parameters: {depth_trainable_params:,}
                Frozen Parameters: {depth_total_params - depth_trainable_params:,}

                Input Shape: [batch_size, 3, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
                """)
            
            # Determine if single-scale or multi-scale based on output type
            is_multi_scale = isinstance(dummy_output, dict)
            
            if is_multi_scale:
                depth_stats += "\nOutput Shapes (Multi-Scale):\n"
                for key in sorted(dummy_output.keys()):
                    shape = dummy_output[key].shape
                    depth_stats += f"  {key}: [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
            else:
                shape = dummy_output.shape
                depth_stats += f"Output Shape (Single-Scale): [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
            
            self.writers['train'].add_text('depth_model_stats', depth_stats)
        
        # Pose Model Statistics
        pose_total_params, pose_trainable_params = count_parameters(self.models["pose_model"])
        
        # Dynamically determine pose model output shapes by forward pass with dummy input
        num_input_channels = 3 * self.num_pose_frames  # 3 channels per frame
        with torch.no_grad():
            dummy_pose_input = torch.randn(1, num_input_channels, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
            dummy_pose_output = self.models["pose_model"](dummy_pose_input)
        
        pose_stats = textwrap.dedent(f"""
            Pose Model Statistics:
            ----------------------
            Model Type: {self.conf['pose_model_type']}
            Total Parameters: {pose_total_params:,}
            Trainable Parameters: {pose_trainable_params:,}
            Frozen Parameters: {pose_total_params - pose_trainable_params:,}

            Input Shape: [batch_size, {num_input_channels}, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
              (Number of input frames: {self.num_pose_frames})
              (Frame IDs for training: {self.conf['frame_ids_training']})
            
            Output Shapes:
            """)
        
        # Handle different pose model output formats
        if isinstance(dummy_pose_output, dict):
            for key in sorted(dummy_pose_output.keys()):
                shape = dummy_pose_output[key].shape
                shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
                pose_stats += f"  {key}: {shape_str}\n"
        elif isinstance(dummy_pose_output, (list, tuple)):
            for i, output in enumerate(dummy_pose_output):
                shape = output.shape
                output_name = "axisangle" if i == 0 else "translation"
                shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
                pose_stats += f"  {output_name}: {shape_str}\n"
        else:
            shape = dummy_pose_output.shape
            shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
            pose_stats += f"  output: {shape_str}\n"
        
        self.writers['train'].add_text('pose_model_stats', pose_stats)
    
    def compute_jepa_loss(self, imgs, masks_enc, masks_pred):
        """
        Compute JEPA masked prediction loss.
        
        Args:
            imgs: Input images [B, 3, H, W]
            masks_enc: Context masks for encoder
            masks_pred: Target masks for prediction
            
        Returns:
            JEPA loss (scalar tensor)
        """
        B = imgs.shape[0]
        
        # Target branch (frozen target encoder)
        with torch.no_grad():
            # Forward through target encoder
            h = self.models["target_encoder"].get_intermediate_layers(imgs, n=1)[-1]
          
            # Normalize over feature dimension
            h = F.layer_norm(h, (h.size(-1),))
         
            # Apply target masks to get regions to predict
            h = apply_masks(h, masks_pred)
            
            # Repeat for each context mask
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

        # Context branch (trainable encoder + predictor)
        # Forward through context encoder
        z = self.models["context_encoder"].get_intermediate_layers(imgs, masks_enc[0], n=1)[-1]
        
        # Forward through predictor to predict target regions
        z = self.models["predictor"](z, masks_enc, masks_pred)
        
        # Compute smooth L1 loss between predictions and targets
        loss = F.smooth_l1_loss(z, h)
        
        return loss
    
    def update_target_encoder(self):
        """
        Update target encoder using EMA (Exponential Moving Average).
        
        θ_target = m * θ_target + (1-m) * θ_context
        """
        with torch.no_grad():
            m = next(self.ema_scheduler)
            for param_context, param_target in zip(
                self.models["context_encoder"].parameters(),
                self.models["target_encoder"].parameters()
            ):
                param_target.data.mul_(m).add_((1. - m) * param_context.detach().data)

        return m  # return current EMA momentum for logging

    def save_model(self):
        """
            Save model weights to disk.
        """
        save_folder = os.path.join(self.log_path, "models", "weights_epoch_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

if __name__ == "__main__":
    lt.monkey_patch()

    conf = Conf().conf  

    trainer = Trainer(conf)
    trainer.train()

