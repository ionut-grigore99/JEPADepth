import os
import glob
import cv2
import numpy as np
import lovely_tensors as lt
import PIL.Image as pil
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
import torch
from torchvision import transforms

from src.models.dinov3.dino import DINODepth
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.datasets.cityscapes_dataset import CityscapesDataset
from src.utils import disp_to_depth, count_parameters, format_number, numpy_to_base64, readlines
from src.config.conf import Conf

# ---------------------------------------------------------------------------
# Grid layout rules:  1 model → (rows=1, cols=2)
#                     3 models → (rows=2, cols=2)
#                     5 models → (rows=2, cols=3)
# ---------------------------------------------------------------------------
GRID_LAYOUTS = {1: (1, 2), 3: (2, 2), 5: (2, 3)}

DATASET_NAME = "KITTI"  # dataset to perform inference on
NUM_IMAGES   = None      # how many images to run inference on (for the report) - set to None to use all test images

# Visual constants for the composite image
BG_COLOR      = (18, 18, 18)    # near-black background
TEXT_COLOR    = (255, 255, 255)
CORNER_FONT_SIZE = 14           # change this to adjust the label font size on images


# ---------------------------------------------------------------------------
# Hardcoded list of models to compare
# Each entry: dict with keys  'display_name', 'model_name', and model-specific config
# ---------------------------------------------------------------------------
MODELS_TO_COMPARE = [
    {
        "display_name": "MonoDepth2",
        "model_name": "monodepth2",
        "num_layers": 18,
        "pretrained": True,
        "encoder_weights_path": "weights/monodepth2/mono_640x192/encoder.pth",
        "decoder_weights_path": "weights/monodepth2/mono_640x192/depth.pth",
        "scales": [0, 1, 2, 3]
    },
    {
        # "display_name": "DINOv3 (w/o JEPA)",
        "display_name": "JEPADepth (ours)",
        "model_name": "dino",
        "encoder_size": "small",
        "decoder_channels": 256,
        "encoder_weights_path": None,
        "decoder_weights_path": None,
        "weights_path": "tensorboard/train/dino/20260228-224926/models/weights_epoch_19/depth_model.pth",
        "scales": [0, 1, 2, 3]
    },
    # {
    #     # "display_name": "DINOv3 (w/o JEPA)",
    #     "display_name": "JEPADepth (ours)",
    #     "model_name": "monovit",
    #     "encoder_weights_path": "weights/monovit/encoder.pth",
    #     "decoder_weights_path": "weights/monovit/depth.pth",
    # },
    {
        # "display_name": "JEPADepth (ours)",
        "display_name": "DINOv3 (w/o JEPA)",
        "model_name": "dino",
        "encoder_size": "small",
        "decoder_channels": 256,
        "encoder_weights_path": "tensorboard/train/jepa_small/20260220-112633/models/weights_epoch_19/context_encoder.pth",
        "decoder_weights_path": "tensorboard/train/jepa_small/20260220-112633/models/weights_epoch_19/depth_decoder.pth",
        "weights_path": None,
        "scales": [0, 1, 2, 3]
    },
    # Add more models here following the same pattern
]

class Make3DInferenceDataset(torch.utils.data.Dataset):
    """Minimal wrapper so Make3D can be iterated like KITTI/Cityscapes."""
    def __init__(self, images, filenames, h, w):
        self.images    = images
        self.filenames = filenames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_np  = self.images[idx].astype(np.uint8)          # [H, W, 3] RGB
        tensor  = self.transform(img_np)                       # [3, h, w]
        return {
            ("color", 0, 0): tensor,
            "filename":       self.filenames[idx],
        }

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load DejaVu Bold; fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def overlay_model_name(img_np: np.ndarray, name: str) -> np.ndarray:
    """Overlay label as white text with black shadow in the top-left corner.
    Supports multi-line text (lines separated by newline).
    Font size is fixed at CORNER_FONT_SIZE so all labels look the same size.
    """
    img_pil = pil.fromarray(img_np)
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(CORNER_FONT_SIZE)
    margin = 10
    line_gap = 4
    x, y = margin, margin
    for line in name.split("\n"):
        # Shadow
        draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return np.array(img_pil)


def build_composite_image(
    rgb_pil: pil.Image,
    depth_arrays: list,      # list of uint8 np arrays (already visualised, label in corner)
    model_names: list,
    model_cfgs: list,
    n_images_total: int,
    filename: str,
) -> pil.Image:
    """
    Stitch RGB + depth predictions into a single clean image (no header banners).
    Labels are burned into the bottom-right corner of each cell.

    Layout (grid cells numbered):
        1 model  →  1 row × 2 cols  :  [RGB | depth1]
        3 models →  2 rows × 2 cols :  [RGB | depth1]
                                       [depth2 | depth3]
        5 models →  2 rows × 3 cols :  [RGB | depth1 | depth2]
                                       [depth3 | depth4 | depth5]
    """
    n_models = len(depth_arrays)
    if n_models not in GRID_LAYOUTS:
        raise ValueError(
            f"Only 1, 3, or 5 models are supported (got {n_models}). "
            f"Check MODELS_TO_COMPARE."
        )
    rows, cols = GRID_LAYOUTS[n_models]

    orig_w, orig_h = rgb_pil.size
    cell_w, cell_h = orig_w, orig_h

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = pil.new("RGB", (canvas_w, canvas_h), color=BG_COLOR)

    # ---- build ordered cell list: RGB first, then depths ----
    cells = [(rgb_pil, "RGB")] + [(pil.fromarray(arr), name) for arr, name in zip(depth_arrays, model_names)]

    # ---- paste cells into the grid ----
    for idx, (img, label) in enumerate(cells):
        row = idx // cols
        col = idx % cols
        x_off = col * cell_w
        y_off = row * cell_h
        # Resize first so all images are the same size before burning the label
        img_resized = img.resize((cell_w, cell_h), pil.Resampling.LANCZOS)
        # Burn label at identical size for every cell
        img_labeled = overlay_model_name(np.array(img_resized), label)
        canvas.paste(pil.fromarray(img_labeled), (x_off, y_off))

    return canvas


def load_model(model_cfg: dict, device: torch.device):
    """Load a single model from its config dict."""
    name = model_cfg["model_name"]
    if name.startswith("dino"):
        model = DINODepth(
            model_cfg["encoder_size"],
            model_cfg["decoder_channels"],
            model_cfg["scales"],
        )
        model.from_pretrained(
            encoder_weights_path=model_cfg["encoder_weights_path"],
            decoder_weights_path=model_cfg["decoder_weights_path"],
            weights_path=model_cfg["weights_path"],
            device=device,
        )
    elif name == "monodepth2":
        model = MonoDepth2(
            num_layers=model_cfg["num_layers"],
            pretrained=model_cfg["pretrained"],
            scales=model_cfg["scales"],
        )
        model.from_pretrained(
            encoder_weights_path=model_cfg["encoder_weights_path"],
            decoder_weights_path=model_cfg["decoder_weights_path"],
            device=device,
        )
    elif name == "monovit":
        from src.models.monovit.monovit import MonoViT
        model = MonoViT()
        model.from_pretrained(
            encoder_weights_path=model_cfg["encoder_weights_path"],
            decoder_weights_path=model_cfg["decoder_weights_path"],
            device=device,
        )
    else:
        raise NotImplementedError(f"Model '{name}' not implemented!")
    model.to(device)
    model.eval()
    return model


def depth_np_to_viz(depth_np: np.ndarray) -> np.ndarray:
    """Convert a raw depth numpy array to a uint8 plasma RGB image."""
    vmax = np.percentile(depth_np, 95)
    normalizer = mpl.colors.Normalize(vmin=depth_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap="plasma")
    return (mapper.to_rgba(depth_np)[:, :, :3] * 255).astype(np.uint8)


def create_html_report(results, html_directory, conf, model_cfgs):
    """Create a comprehensive HTML report — one composite image per input."""

    results_html = ""
    for idx, result in enumerate(results):
        filename   = result["filename"]
        composite_b64 = result["composite_b64"]

        results_html += f"""
        <div class="result-item">
            <div class="result-header">
                <h3>📸 {filename}</h3>
                <div class="index">#{idx + 1}</div>
            </div>
            <div style="padding:20px 30px 30px;">
                <img src="{composite_b64}" alt="Comparison grid for {filename}"
                     class="clickable-image composite-img"
                     onclick="openLightbox(this.src, '{filename}')">
            </div>
        </div>
        """

    # Build model list rows for the dedicated "Models" section
    model_rows_html = ""
    for i, cfg in enumerate(model_cfgs):
        param_str = format_number(cfg["_params"]) if "_params" in cfg else "—"
        model_rows_html += f"""
        <tr>
            <td class="mnum">{i + 1}</td>
            <td class="mname">{cfg['display_name']}</td>
            <td class="mparams">{param_str}</td>
        </tr>"""

    n_models   = len(model_cfgs)
    rows, cols = GRID_LAYOUTS.get(n_models, (1, n_models + 1))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Depth Estimation — Models Comparison</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family:'Segoe UI',sans-serif; background:linear-gradient(135deg,#667eea,#764ba2); padding:20px; color:#333; }}
        .container {{ max-width:1600px; margin:0 auto; background:white; border-radius:15px; box-shadow:0 20px 60px rgba(0,0,0,.3); overflow:hidden; }}
        .header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:40px; text-align:center; }}
        .header h1 {{ font-size:2.2em; text-shadow:2px 2px 4px rgba(0,0,0,.2); }}
        .header p  {{ margin-top:8px; opacity:.85; font-size:1em; }}
        /* Run meta strip */
        .run-meta {{ display:flex; gap:30px; flex-wrap:wrap; padding:22px 30px; background:#f8f9fa; border-bottom:3px solid #667eea; align-items:center; }}
        .meta-chip {{ background:white; padding:10px 20px; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,.1); text-align:center; }}
        .meta-chip .val {{ font-size:1.3em; font-weight:bold; color:#667eea; }}
        .meta-chip .lbl {{ font-size:.8em; color:#666; text-transform:uppercase; letter-spacing:1px; margin-top:2px; }}
        /* Models table */
        .models-section {{ padding:28px 30px 10px; border-bottom:2px solid #e8ecf8; }}
        .models-section h2 {{ color:#667eea; font-size:1.3em; margin-bottom:14px; }}
        .models-table {{ width:100%; border-collapse:collapse; font-size:.95em; }}
        .models-table th {{ background:#667eea; color:white; padding:10px 16px; text-align:left; font-weight:600; }}
        .models-table td {{ padding:9px 16px; border-bottom:1px solid #eef0f8; }}
        .models-table tr:last-child td {{ border-bottom:none; }}
        .models-table tr:hover td {{ background:#f4f5ff; }}
        .mnum  {{ width:40px; color:#999; font-size:.85em; }}
        .mname {{ font-weight:600; color:#333; }}
        .mparams {{ color:#555; }}
        /* Results */
        .results-section {{ padding:30px; }}
        .results-section h2 {{ color:#667eea; margin-bottom:30px; padding-bottom:10px; border-bottom:2px solid #667eea; }}
        .result-item {{ margin-bottom:50px; background:#f8f9fa; border-radius:15px; overflow:hidden; box-shadow:0 4px 6px rgba(0,0,0,.1); }}
        .result-header {{ background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:20px 30px; display:flex; justify-content:space-between; align-items:center; }}
        .result-header h3 {{ font-size:1.2em; margin:0; }}
        .result-header .index {{ background:rgba(255,255,255,.2); padding:5px 15px; border-radius:20px; font-size:.9em; }}
        .composite-img {{ width:100%; height:auto; display:block; border-radius:8px; }}
        .clickable-image {{ cursor:zoom-in; transition:transform .3s,box-shadow .3s; }}
        .clickable-image:hover {{ transform:scale(1.01); box-shadow:0 8px 20px rgba(0,0,0,.35); }}
        /* Lightbox */
        .lightbox-modal {{ display:none; position:fixed; z-index:9999; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,.95); cursor:zoom-out; overflow:auto; }}
        .lightbox-modal.active {{ display:flex; justify-content:center; align-items:flex-start; padding:20px; }}
        .lightbox-content {{ max-width:100%; height:auto; object-fit:contain; }}
        .lightbox-close {{ position:fixed; top:20px; right:40px; color:white; font-size:50px; cursor:pointer; z-index:10000; }}
        .lightbox-caption {{ position:fixed; bottom:20px; left:50%; transform:translateX(-50%); color:white; background:rgba(0,0,0,.7); padding:12px 28px; border-radius:10px; }}
    </style>
</head>
<body>
    <!-- Lightbox -->
    <div id="lightbox" class="lightbox-modal" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <img class="lightbox-content" id="lightbox-img">
        <div class="lightbox-caption" id="lightbox-caption"></div>
    </div>
    <script>
        function openLightbox(src, cap) {{
            document.getElementById('lightbox').classList.add('active');
            document.getElementById('lightbox-img').src = src;
            document.getElementById('lightbox-caption').textContent = cap;
            document.body.style.overflow = 'hidden';
        }}
        function closeLightbox() {{
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = 'auto';
        }}
        document.addEventListener('keydown', e => {{ if(e.key==='Escape') closeLightbox(); }});
        document.getElementById('lightbox-img').addEventListener('click', e => e.stopPropagation());
    </script>

    <div class="container">
        <div class="header">
            <h1>🎯 Monocular Depth Estimation — Models Comparison</h1>
        </div>

        <!-- Run meta chips -->
        <div class="run-meta">
            <div class="meta-chip">
                <div class="val">{len(results)}</div>
                <div class="lbl">Images</div>
            </div>
            <div class="meta-chip">
                <div class="val">{n_models}</div>
                <div class="lbl">Models</div>
            </div>
            <div class="meta-chip">
                <div class="val">{conf['im_sz'][1]}×{conf['im_sz'][0]}</div>
                <div class="lbl">Input resolution</div>
            </div>
            <div class="meta-chip">
                <div class="val">{DATASET_NAME}</div>
                <div class="lbl">Dataset used</div>
            </div>
        </div>

        <!-- Models list — shown BEFORE any images -->
        <div class="models-section">
            <h2>🤖 Models compared</h2>
            <table class="models-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Name</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {model_rows_html}
                </tbody>
            </table>
        </div>

        <!-- Composite images -->
        <div class="results-section">
            <h2>🖼️ Depth Predictions</h2>
            {results_html}
        </div>

    </div>
</body>
</html>"""

    os.makedirs(html_directory, exist_ok=True)
    report_path = os.path.join(html_directory, "inference_models_comparison.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   HTML report saved to: {report_path}")
    return report_path


def models_comparison(conf):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    h = conf["im_sz"][0]
    w = conf["im_sz"][1]

    # Load ALL models upfront
    print("\n-> Loading models...")
    loaded_models = []
    for cfg in MODELS_TO_COMPARE:
        print(f"   Loading: {cfg['display_name']}")
        model = load_model(cfg, device)
        n_params, _ = count_parameters(model)
        cfg["_params"] = n_params          # store for HTML report + image labels
        print(f"   Params:  {format_number(n_params)}")
        loaded_models.append((cfg, model))

    # Collect images 
    if DATASET_NAME == "KITTI":
        splits_dir = conf['data_path'].replace('kitti_data', 'kitti_splits')
        filenames = readlines(os.path.join(splits_dir, conf['evaluation_split'], "test_files.txt"))
        dataset = KITTIRAWDataset(conf['data_path'], filenames, conf['im_sz'][0], conf['im_sz'][1], [0], 4, is_train=False, img_ext='.png' if conf['train_from_png'] else '.jpg')
    elif DATASET_NAME == "Cityscapes":
        filenames = readlines(os.path.join(conf['cityscapes_dataset_path'], "cityscapes_test_files.txt"))
        dataset = CityscapesDataset(conf['cityscapes_dataset_path'], filenames, conf['im_sz'][0], conf['im_sz'][1], [0], 4, is_train=False, img_ext='.png' if conf['train_from_png'] else '.jpg')
    elif DATASET_NAME == "Make3D":
        with open(os.path.join(conf['make3d_dataset_path'], "make3d_test_files.txt")) as f:
            test_filenames = [x[4:] for x in f.read().splitlines()]
        color_new_height = int(1704 / 2)
        images, filenames = [], []
        for fname in test_filenames:
            img = cv2.imread(os.path.join(conf['make3d_dataset_path'], "Test134", f"img-{fname}.jpg"))
            img = img[int((2272 - color_new_height) / 2):int((2272 + color_new_height) / 2), :, :]
            images.append(img[:, :, ::-1])   # BGR -> RGB
            filenames.append(fname)
        dataset = Make3DInferenceDataset(images, filenames, h, w)
    else:
        raise ValueError(f"Unknown DATASET_NAME '{DATASET_NAME}'. Check the code for supported datasets.")
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=conf['num_workers'], pin_memory=True, drop_last=False)
    print(f"\n-> Predicting on {NUM_IMAGES} image(s) with {len(loaded_models)} model(s)")

    # Inference loop
    results = []

    with torch.no_grad():
        for img_idx, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].to(device)  
            filename = filenames[img_idx] 

            # Reconstruct PIL RGB for composite image (denormalize from [0,1])
            rgb_np  = (input_color.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_pil = pil.fromarray(rgb_np)

            depth_arrays, model_names = [], []

            for cfg, model in loaded_models:
                out        = model(input_color)
                norm_disp  = out[("disp", 0)]
                disp, depth   = disp_to_depth(norm_disp, conf["min_depth"], conf["max_depth"])
                viz_np   = disp.squeeze().cpu().numpy()
                depth_viz  = depth_np_to_viz(viz_np)

                depth_arrays.append(depth_viz)
                model_names.append(cfg["display_name"])

            # Build composite image (RGB + all depth predictions stitched)
            composite_pil = build_composite_image(
                rgb_pil      = rgb_pil,
                depth_arrays = depth_arrays,
                model_names  = model_names,
                model_cfgs   = [cfg for cfg, _ in loaded_models],
                n_images_total = len(dataloader),
                filename     = filename,
            )

            results.append({
                "filename":      filename,
                "composite_b64": numpy_to_base64(np.array(composite_pil)),
            })

            print(f"   [{img_idx + 1}/{len(dataloader)}] {filename}")

            if NUM_IMAGES is not None and img_idx + 1 >= NUM_IMAGES:
                break

    # Generate HTML
    print("\n-> Generating HTML report...")
    create_html_report(results, conf["htmls_path"], conf, MODELS_TO_COMPARE)
    print("\n-> Done!")


if __name__ == "__main__":
    lt.monkey_patch()
    conf = Conf().conf
    models_comparison(conf)