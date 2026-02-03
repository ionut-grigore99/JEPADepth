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
from src.models.layers import *
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *

# JEPA imports
from src.masks.mask_collator import MaskCollator as MBMaskCollator
from src.utils import apply_masks, repeat_interleave_batch, trunc_normal_
import src.models.ijepa.vision_transformer as vit
from src.ijepa.utils.logging import AverageMeter

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
            self.models["depth_model"].from_pretrained(weights_path=self.conf['pixio']['weights_path'], device=self.device) if self.conf['load_pretrained_depth_model'] else None
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
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.conf['loss_scales']: # default we have just 1 scale
            h = self.conf['im_sz'][0] // (2 ** scale) 
            w = self.conf['im_sz'][1] // (2 ** scale) 

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.conf['bs'], h, w)
            self.backproject_depth[scale].to(self.device)


            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.conf['bs'], h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        # JEPA-style training components
        self.use_jepa = self.conf['use_jepa_training']
        
        if self.use_jepa:
            print("Initializing JEPA-style training components...")
            
            # JEPA config parameters
            jepa_conf = self.conf['jepa']
            self.jepa_weight = jepa_conf['loss_weight']
            self.use_bfloat16 = jepa_conf['use_bfloat16']

            # Mask collator for creating context and target masks
            patch_size = jepa_conf['patch_size']
            pred_mask_scale = jepa_conf['pred_mask_scale']
            enc_mask_scale = jepa_conf['enc_mask_scale']
            aspect_ratio = jepa_conf['aspect_ratio']
            num_enc_masks = jepa_conf['num_enc_masks']
            num_pred_masks = jepa_conf['num_pred_masks']
            allow_overlap = jepa_conf['allow_overlap']
            min_keep = jepa_conf['min_keep']

            self.mask_collator = MBMaskCollator(
                input_size=(self.conf['im_sz'][0], self.conf['im_sz'][1]),
                patch_size=patch_size,
                pred_mask_scale=pred_mask_scale,
                enc_mask_scale=enc_mask_scale,
                aspect_ratio=aspect_ratio,
                nenc=num_enc_masks,
                npred=num_pred_masks,
                allow_overlap=allow_overlap,
                min_keep=min_keep
            )
            
            # Target encoder (EMA copy of depth encoder's backbone)
            # We create a frozen copy that will be updated with momentum
            self.models["target_encoder"] = copy.deepcopy(self.models["depth_model"].encoder)
            for p in self.models["target_encoder"].parameters():
                p.requires_grad = False
            self.models["target_encoder"].to(self.device)
            
            # Predictor network for JEPA
            pred_depth = jepa_conf.get('predictor_depth', 6)
            pred_emb_dim = jepa_conf.get('predictor_emb_dim', 384)
            
            # Get encoder embedding dimension from the depth model's encoder
            if hasattr(self.models["depth_model"].encoder, 'embed_dim'):
                encoder_embed_dim = self.models["depth_model"].encoder.embed_dim
            else:
                encoder_embed_dim = 768  # default for ViT-Base
            
            # Calculate number of patches for the predictor
            num_patches = (self.conf['im_sz'][0] // patch_size) * (self.conf['im_sz'][1] // patch_size)
            
            self.models["predictor"] = vit.vit_predictor(
                num_patches=num_patches,
                embed_dim=encoder_embed_dim,
                predictor_embed_dim=pred_emb_dim,
                depth=pred_depth,
                num_heads=encoder_embed_dim // 64  # typical head dimension is 64
            )
            
            # Initialize predictor weights
            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.constant_(m.bias, 0)
                    torch.nn.init.constant_(m.weight, 1.0)
            
            for m in self.models["predictor"].modules():
                init_weights(m)
            
            self.models["predictor"].to(self.device)
            self.parameters_to_train += list(self.models["predictor"].parameters())
            
            # Re-initialize optimizer to include predictor parameters
            self.model_optimizer = optim.AdamW(
                self.parameters_to_train, 
                self.conf['learning_rate'], 
                weight_decay=self.conf['weight_decay']
            )
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, 
                self.conf['scheduler_step_size'], 
                0.1
            )
            
            # Momentum scheduler for EMA updates
            ema_start = jepa_conf.get('ema', [0.996, 1.0])[0]
            ema_end = jepa_conf.get('ema', [0.996, 1.0])[1]
            ipe = len(self.train_loader)
            self.momentum_scheduler = (
                ema_start + i * (ema_end - ema_start) / (ipe * self.conf['num_epochs'])
                for i in range(int(ipe * self.conf['num_epochs']) + 1)
            )
            
            # JEPA loss meters
            self.jepa_loss_meter = AverageMeter()
            
            # Scaler for mixed precision training
            if self.use_bfloat16:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None

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
            
            if self.conf['clip_grad_norm'] is not None: 
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, self.conf['clip_grad_norm'])
            self.model_optimizer.step()
            
            # Update target encoder with momentum (JEPA-specific)
            if self.use_jepa:
                self.update_target_encoder()

            duration_optimization = time.time() - before_optimization_time

            # log less frequently after the first 2000 steps to save time & disk space:
            #  - log every 10 batches if step < 2000, otherwise log every 1000 steps
            #  - 1 step = 1 iteration
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
            If JEPA training is enabled, also compute JEPA masked prediction loss.
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

        # JEPA loss computation
        if self.use_jepa:
            jepa_loss = self.compute_jepa_loss(inputs["color_aug", 0, 0])
            losses["jepa_loss"] = jepa_loss
            losses["loss"] = losses["loss"] + self.jepa_weight * jepa_loss
            self.jepa_loss_meter.update(jepa_loss.item())

        return outputs_dict, losses
    
    def compute_jepa_loss(self, imgs):
        """
            Compute JEPA masked prediction loss.
            
            Note: This is a simplified implementation that works with the Pixio encoder.
            The Pixio encoder doesn't natively support masked input, so we:
            1. Extract full embeddings from both encoders
            2. Apply masks to the embeddings post-hoc
            3. Predict masked regions using the predictor
            
            Args:
                imgs: Input images [B, 3, H, W]
            
            Returns:
                jepa_loss: Smooth L1 loss between predicted and target embeddings
        """
        B = imgs.shape[0]
        h_patches = self.conf['im_sz'][0] // 16
        w_patches = self.conf['im_sz'][1] // 16
        num_patches = h_patches * w_patches
        
        # Step 1: Generate masks (simplified version)
        # In full JEPA, we'd use the mask_collator, but for compatibility we use simple random masks
        masks_enc = []
        masks_pred = []
        
        for _ in range(B):
            # Context mask: keep ~90% of patches
            num_keep_enc = int(num_patches * 0.9)
            mask_enc = torch.randperm(num_patches, device=self.device)[:num_keep_enc]
            
            # Target mask: predict ~15% of patches (non-overlapping with context)
            remaining = torch.randperm(num_patches, device=self.device)[num_keep_enc:]
            num_keep_pred = min(int(num_patches * 0.15), len(remaining))
            mask_pred = remaining[:num_keep_pred]
            
            masks_enc.append([mask_enc])
            masks_pred.append([mask_pred])
        
        # Step 2: Forward through target encoder (frozen, no gradient)
        with torch.no_grad():
            # Get full embeddings from target encoder
            # The Pixio encoder returns a list of features from different blocks
            target_features = self.models["target_encoder"](imgs)
            
            # Use the last block's patch tokens as our target embeddings
            # Shape: [B, num_patches, embed_dim]
            h = target_features[-1]['patch_tokens_norm']
            
            # Normalize features (JEPA uses layer normalization)
            h = F.layer_norm(h, (h.size(-1),))
            
            # Apply target masks to select which regions to predict
            h = apply_masks(h, masks_pred)
            
            # Repeat for each context mask (typically just 1)
            h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        
        # Step 3: Forward through context encoder (trainable, with gradients)
        # Get embeddings from the trainable encoder
        context_features = self.models["depth_model"].encoder(imgs)
        z = context_features[-1]['patch_tokens_norm']
        
        # Apply context masks
        z = apply_masks(z, masks_enc)
        
        # Step 4: Predict target embeddings from context using predictor
        z = self.models["predictor"](z, masks_enc, masks_pred)
        
        # Step 5: Compute loss between predictions and targets
        jepa_loss = F.smooth_l1_loss(z, h)
        
        return jepa_loss
    
    def update_target_encoder(self):
        """
            Momentum update of target encoder.
            Uses exponential moving average (EMA) to update target encoder parameters.
        """
        if self.use_jepa:
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(
                    self.models["depth_model"].encoder.parameters(),
                    self.models["target_encoder"].parameters()
                ):
                    param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

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
        depth_total_params, depth_trainable_params = count_parameters(self.models["depth_model"])
        
        # Dynamically determine output shapes by forward pass with dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
            dummy_output = self.models["depth_model"](dummy_input)
        
        # Determine if single-scale or multi-scale based on output type
        is_multi_scale = isinstance(dummy_output, dict)
        
        depth_stats = textwrap.dedent(f"""
            Depth Model Statistics:
            -----------------------
            Model Type: {self.conf['model_name']}
            Total Parameters: {depth_total_params:,}
            Trainable Parameters: {depth_trainable_params:,}
            Frozen Parameters: {depth_total_params - depth_trainable_params:,}

            Input Shape: [batch_size, 3, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
            """)
        
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
        
        # JEPA Components Statistics (if enabled)
        if self.use_jepa:
            target_encoder_total_params, target_encoder_trainable_params = count_parameters(self.models["target_encoder"])
            predictor_total_params, predictor_trainable_params = count_parameters(self.models["predictor"])

            jepa_stats = textwrap.dedent(f"""
                JEPA Components Statistics:
                ----------------------------
                Target Encoder (frozen EMA copy):
                  Total Parameters: {target_encoder_total_params:,}
                  Trainable Parameters: {target_encoder_trainable_params:,}
                  Frozen Parameters: {target_encoder_total_params - target_encoder_trainable_params:,}

                Predictor Network:
                  Total Parameters: {predictor_total_params:,}
                  Trainable Parameters: {predictor_trainable_params:,}
                  Frozen Parameters: {predictor_total_params - predictor_trainable_params:,}
                """)
            
            self.writers['train'].add_text('jepa_stats', jepa_stats)

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

