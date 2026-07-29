"""
Find (automatically) the KITTI test images where JEPADepth beats DINOv3-without-JEPA, instead of visually scanning hundreds of composite grids.

Idea
----
The Eigen test split has ground-truth depth, so we can *score* every image rather than eyeball it. For each test image we:
  1. run both models,
  2. compute per-image depth metrics using the EXACT same protocol as `src/evaluation/kitti_evaluation.py` (per-image median scaling, Garg/Eigen crop, clamp to [1e-3, 80]),
  3. take the per-image improvement of JEPADepth over the DINO baseline,
  4. rank all images by that improvement.

Outputs:
  - `jepa_wins_ranking.csv`   : every image with both models' metrics + deltas, sorted best-win first.
  - `jepa_wins_topk.html`     : a gallery of the TOP_K biggest wins (RGB | DINO | JEPADepth), each cell annotated with its AbsRel and the improvement.
"""

import os
import csv
import cv2
import numpy as np
import torch
import lovely_tensors as lt
from tqdm import tqdm
from torch.utils.data import DataLoader
import PIL.Image as pil

from src.config.conf import Conf
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.utils import disp_to_depth, readlines, numpy_to_base64, format_number, count_parameters
from src.evaluation.utils import compute_errors
from src.analysis.inference_models_comparison import load_model, depth_np_to_viz, overlay_model_name

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K        = 30           # how many of the biggest wins to put in the HTML gallery
RANK_METRIC  = "abs_rel"    # metric used to rank wins; choices: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
RANK_BY      = "absolute"   # "absolute" delta, or "relative" (% improvement vs baseline)
MIN_VALID_PX = 50           # skip images with too few valid GT pixels to score reliably

METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
LOWER_BETTER = {"abs_rel": True, "sq_rel": True, "rmse": True, "rmse_log": True,
                "a1": False, "a2": False, "a3": False}

MIN_DEPTH_EVAL = 1e-3
MAX_DEPTH_EVAL = 80

# The two models to compare. BASELINE is the reference, OURS is what we hope wins.
BASELINE_CFG = {
    "display_name": "DINOv3 (w/o JEPA)",
    "model_name": "dino",
    "encoder_size": "small",
    "decoder_channels": 256,
    "encoder_weights_path": None,
    "decoder_weights_path": None,
    "weights_path": "tensorboard/train/dino/20260228-224926/models/weights_epoch_19/depth_model.pth",
    "scales": [0, 1, 2, 3],
    "decoder_type": "fpn",
    "use_lora": False,
    "lora_rank": 16,
}
OURS_CFG = {
    "display_name": "JEPADepth (ours)",
    "model_name": "dino",
    "encoder_size": "small",
    "decoder_channels": 256,
    "encoder_weights_path": "tensorboard/train/jepa_small/20260220-112633/models/weights_epoch_19/context_encoder.pth",
    "decoder_weights_path": "tensorboard/train/jepa_small/20260220-112633/models/weights_epoch_19/depth_decoder.pth",
    "weights_path": None,
    "scales": [0, 1, 2, 3],
    "decoder_type": "fpn",
    "use_lora": False,
    "lora_rank": 16,
}


def per_image_errors(pred_disp_model_res, gt_depth, split):
    """Replicates the per-image scoring in kitti_evaluation.evaluate().

    Returns (dict metric_name->value, n_valid_pixels). Returns (None, n) if not scorable.
    """
    gt_h, gt_w = gt_depth.shape[:2]
    pred_disp = cv2.resize(pred_disp_model_res, (gt_w, gt_h))
    pred_depth = 1.0 / pred_disp

    if split == "eigen":
        mask = np.logical_and(gt_depth > MIN_DEPTH_EVAL, gt_depth < MAX_DEPTH_EVAL)
        crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h,
                         0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    else:
        mask = gt_depth > 0

    n_valid = int(mask.sum())
    if n_valid < MIN_VALID_PX:
        return None, n_valid

    pred_depth = pred_depth[mask]
    gt = gt_depth[mask]

    # per-image median scaling (mono protocol)
    ratio = np.median(gt) / np.median(pred_depth)
    pred_depth = pred_depth * ratio

    pred_depth[pred_depth < MIN_DEPTH_EVAL] = MIN_DEPTH_EVAL
    pred_depth[pred_depth > MAX_DEPTH_EVAL] = MAX_DEPTH_EVAL

    errs = compute_errors(gt, pred_depth)   # (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    return dict(zip(METRIC_NAMES, [float(e) for e in errs])), n_valid


@torch.no_grad()
def infer_disp(model, input_color, conf):
    """Return the model-resolution disparity map [H, W] as numpy (same as kitti eval)."""
    norm_disp = model(input_color)[("disp", 0)]
    pred_disp, _ = disp_to_depth(norm_disp, conf["min_depth"], conf["max_depth"])
    return pred_disp.cpu()[0, 0].numpy()


def improvement(row):
    """Signed improvement of OURS over BASELINE on RANK_METRIC (positive = ours better)."""
    b = row["baseline"][RANK_METRIC]
    o = row["ours"][RANK_METRIC]
    diff = (b - o) if LOWER_BETTER[RANK_METRIC] else (o - b)   # positive => ours better
    if RANK_BY == "relative":
        denom = abs(b) if abs(b) > 1e-9 else 1e-9
        return diff / denom
    return diff


def build_triptych(rgb_uint8, disp_base, disp_ours, row):
    """Stitch [RGB | DINO baseline | JEPADepth] into one labeled row image."""
    viz_base = depth_np_to_viz(disp_base)   # plasma disparity viz (matches the comparison report)
    viz_ours = depth_np_to_viz(disp_ours)

    m = RANK_METRIC
    imp = row["improvement"]
    imp_str = (f"{imp*100:.1f}%" if RANK_BY == "relative" else f"{imp:.3f}")
    cells = [
        (rgb_uint8, "RGB"),
        (viz_base, f"{BASELINE_CFG['display_name']}\n{m}={row['baseline'][m]:.3f}"),
        (viz_ours, f"{OURS_CFG['display_name']}\n{m}={row['ours'][m]:.3f}  (+{imp_str})"),
    ]
    h, w = rgb_uint8.shape[:2]
    canvas = np.full((h, w * 3, 3), 18, dtype=np.uint8)
    for i, (arr, label) in enumerate(cells):
        if arr.shape[:2] != (h, w):
            arr = np.array(pil.fromarray(arr).resize((w, h), pil.Resampling.LANCZOS))
        canvas[:, i * w:(i + 1) * w] = overlay_model_name(arr, label)
    return canvas


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["rank", "index", "filename", f"improvement_{RANK_METRIC}_{RANK_BY}", "n_valid_px"]
        header += [f"baseline_{m}" for m in METRIC_NAMES]
        header += [f"ours_{m}" for m in METRIC_NAMES]
        writer.writerow(header)
        for rank, r in enumerate(rows, 1):
            line = [rank, r["index"], r["filename"], f"{r['improvement']:.6f}", r["n_valid"]]
            line += [f"{r['baseline'][m]:.6f}" for m in METRIC_NAMES]
            line += [f"{r['ours'][m]:.6f}" for m in METRIC_NAMES]
            writer.writerow(line)


def write_html(top_rows, path, n_total, n_wins):
    cards = ""
    for rank, r in enumerate(top_rows, 1):
        cards += f"""
        <div class="card">
          <div class="card-h">
            <span>#{rank} &nbsp; {r['filename']}</span>
            <span class="imp">+{(r['improvement']*100):.1f}%</span>
          </div>
          <img class="zoom" src="{r['composite_b64']}" onclick="lb(this.src)">
        </div>"""
    metric_lbl = f"{RANK_METRIC} ({'relative %' if RANK_BY=='relative' else 'absolute'})"
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>JEPADepth wins vs DINO (w/o JEPA)</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;color:#222}}
.container{{max-width:1500px;margin:0 auto;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.3)}}
.header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:32px 40px;text-align:center}}
.header h1{{font-size:1.9em}}
.meta{{display:flex;gap:24px;flex-wrap:wrap;padding:20px 40px;background:#f6f7fb;border-bottom:3px solid #667eea}}
.chip{{background:#fff;padding:10px 18px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);text-align:center}}
.chip .v{{font-size:1.25em;font-weight:700;color:#667eea}} .chip .l{{font-size:.75em;color:#666;text-transform:uppercase;letter-spacing:1px}}
.grid{{padding:24px 40px 40px}}
.card{{margin-bottom:26px;border:1px solid #e8eaf3;border-radius:10px;overflow:hidden;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.card-h{{display:flex;justify-content:space-between;align-items:center;background:#2b2f45;color:#fff;padding:10px 16px;font-size:.95em}}
.card-h .imp{{background:#2e7d32;padding:3px 12px;border-radius:14px;font-weight:700}}
.card img{{width:100%;display:block;cursor:zoom-in}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.95);z-index:99;cursor:zoom-out;overflow:auto;padding:20px}}
#lb img{{max-width:100%;margin:auto;display:block}}
</style></head><body>
<div id="lb" onclick="this.style.display='none'"><img id="lbimg"></div>
<script>function lb(s){{document.getElementById('lbimg').src=s;document.getElementById('lb').style.display='block'}}</script>
<div class="container">
  <div class="header"><h1>🏆 Where JEPADepth beats DINOv3 (w/o JEPA)</h1>
  <p>Ranked by per-image {metric_lbl} improvement on the KITTI Eigen split (ground-truth scored)</p></div>
  <div class="meta">
    <div class="chip"><div class="v">{n_total}</div><div class="l">Images scored</div></div>
    <div class="chip"><div class="v">{n_wins}</div><div class="l">JEPADepth wins</div></div>
    <div class="chip"><div class="v">{len(top_rows)}</div><div class="l">Shown (top-K)</div></div>
    <div class="chip"><div class="v">{RANK_METRIC}</div><div class="l">Rank metric</div></div>
  </div>
  <div class="grid">{cards}</div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main(conf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = conf["evaluation_split"]
    splits_dir = conf["data_path"].replace("kitti_data", "kitti_splits")

    filenames = readlines(os.path.join(splits_dir, split, "test_files.txt"))
    gt_depths = np.load(os.path.join(splits_dir, split, "gt_depths.npz"),
                        fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    dataset = KITTIRAWDataset(conf["data_path"], filenames, conf["im_sz"][0], conf["im_sz"][1],
                              [0], 4, is_train=False, img_ext=".png" if conf["train_from_png"] else ".jpg")
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=conf["num_workers"], pin_memory=True)

    print("-> Loading models...")
    model_base = load_model(BASELINE_CFG, device)
    model_ours = load_model(OURS_CFG, device)
    print(f"   {BASELINE_CFG['display_name']}: {format_number(count_parameters(model_base)[0])} params")
    print(f"   {OURS_CFG['display_name']}: {format_number(count_parameters(model_ours)[0])} params")

    # ---- Pass 1: score every image ----
    print(f"\n-> Scoring {len(dataset)} images on the '{split}' split...")
    rows = []
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            x = data[("color", 0, 0)].to(device)
            db = infer_disp(model_base, x, conf)
            do = infer_disp(model_ours, x, conf)
            eb, nb = per_image_errors(db, gt_depths[idx], split)
            eo, no = per_image_errors(do, gt_depths[idx], split)
            if eb is None or eo is None:
                continue
            row = {"index": idx, "filename": filenames[idx],
                   "baseline": eb, "ours": eo, "n_valid": min(nb, no)}
            row["improvement"] = improvement(row)
            rows.append(row)

    rows.sort(key=lambda r: r["improvement"], reverse=True)
    n_wins = sum(1 for r in rows if r["improvement"] > 0)

    # ---- Write full ranking CSV ----
    os.makedirs(conf["htmls_path"], exist_ok=True)
    csv_path = os.path.join(conf["htmls_path"], "jepa_wins_ranking.csv")
    write_csv(rows, csv_path)
    print(f"\n-> Ranking CSV: {csv_path}")
    print(f"   JEPADepth wins on {n_wins}/{len(rows)} scored images "
          f"({100*n_wins/max(1,len(rows)):.1f}%) by {RANK_METRIC}.")

    print(f"\n   Top {min(TOP_K,len(rows))} wins ({RANK_METRIC}, {RANK_BY}):")
    for rank, r in enumerate(rows[:min(TOP_K, len(rows))], 1):
        imp = f"{r['improvement']*100:.1f}%" if RANK_BY == "relative" else f"{r['improvement']:.3f}"
        print(f"   {rank:>3}. {r['filename']:<40} "
              f"{RANK_METRIC}: {r['baseline'][RANK_METRIC]:.3f} -> {r['ours'][RANK_METRIC]:.3f}  (+{imp})")

    # ---- Pass 2: build gallery for the top-K wins ----
    top_rows = rows[:TOP_K]
    print(f"\n-> Rendering top-{len(top_rows)} gallery...")
    with torch.no_grad():
        for r in tqdm(top_rows):
            sample = dataset[r["index"]][("color", 0, 0)]
            rgb = (sample.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            x = sample.unsqueeze(0).to(device)
            db = infer_disp(model_base, x, conf)
            do = infer_disp(model_ours, x, conf)
            composite = build_triptych(rgb, db, do, r)
            r["composite_b64"] = numpy_to_base64(composite)

    html_path = os.path.join(conf["htmls_path"], "jepa_wins_topk.html")
    write_html(top_rows, html_path, n_total=len(rows), n_wins=n_wins)
    print(f"-> Gallery: {html_path}")
    print("\n-> Done!")


if __name__ == "__main__":
    lt.monkey_patch()
    conf = Conf().conf
    main(conf)
