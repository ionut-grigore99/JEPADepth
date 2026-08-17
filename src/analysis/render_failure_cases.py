"""
Render 1-2 failure cases for the paper.
The cases are chosen objectively from the per-image ranking produced by `src.analysis.find_jepa_wins` (htmls/jepa_wins_ranking.csv).
Figure layout, one row per case:
    [ Input RGB | DINOv3 (w/o JEPA) | JEPADepth (ours) ]
"""

import os
import numpy as np
import torch
import lovely_tensors as lt
import matplotlib.pyplot as plt

from src.config.conf import Conf
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.utils import readlines
from src.analysis.find_jepa_wins import load_model, infer_disp, per_image_errors, BASELINE_CFG, OURS_CFG
from src.analysis.inference_models_comparison import depth_np_to_viz

FAILURE_INDICES = [383, 602]
OUT_PNG = "assets/jepadepth_failure_cases.png"


def main(conf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = conf["evaluation_split"]
    splits_dir = conf["data_path"].replace("kitti_data", "kitti_splits")

    filenames = readlines(os.path.join(splits_dir, split, "test_files.txt"))
    gt_depths = np.load(os.path.join(splits_dir, split, "gt_depths.npz"), fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    dataset = KITTIRAWDataset(conf["data_path"], filenames, conf["im_sz"][0], conf["im_sz"][1], [0], 4, is_train=False, img_ext=".png" if conf["train_from_png"] else ".jpg")

    print("-> Loading models...")
    model_base = load_model(BASELINE_CFG, device)
    model_ours = load_model(OURS_CFG, device)

    n = len(FAILURE_INDICES)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Input", BASELINE_CFG["display_name"], OURS_CFG["display_name"]]

    for row, idx in enumerate(FAILURE_INDICES):
        sample = dataset[idx][("color", 0, 0)]
        rgb = sample.permute(1, 2, 0).cpu().numpy()
        gt = gt_depths[idx]

        x = sample.unsqueeze(0).to(device)
        with torch.no_grad():
            disp_base = infer_disp(model_base, x, conf)
            disp_ours = infer_disp(model_ours, x, conf)

        err_base, _ = per_image_errors(disp_base, gt, split)
        err_ours, _ = per_image_errors(disp_ours, gt, split)

        panels = [
            (np.clip(rgb, 0, 1), None),
            (depth_np_to_viz(disp_base), f"AbsRel = {err_base['abs_rel']:.3f}"),
            (depth_np_to_viz(disp_ours), f"AbsRel = {err_ours['abs_rel']:.3f}  (worse)"),
        ]

        for col, (img, sublabel) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, fontweight="bold")
            if sublabel:
                ax.set_xlabel(sublabel, fontsize=11, color=("#C62828" if "worse" in sublabel else "#222"))
        axes[row, 0].set_ylabel(os.path.basename(filenames[idx]).replace(" ", "\n"), fontsize=8, rotation=0, ha="right", va="center", labelpad=30)

    fig.suptitle("Failure cases: JEPADepth under-performs the DINOv3 baseline", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"-> Saved figure: {OUT_PNG}")

    print("\n-> Caption-ready numbers:")
    for idx in FAILURE_INDICES:
        gt = gt_depths[idx]
        with torch.no_grad():
            x = dataset[idx][("color", 0, 0)].unsqueeze(0).to(device)
            eb, _ = per_image_errors(infer_disp(model_base, x, conf), gt, split)
            eo, _ = per_image_errors(infer_disp(model_ours, x, conf), gt, split)
        print(f"   idx {idx:>3} ({filenames[idx]}): "
              f"AbsRel {eb['abs_rel']:.3f} (baseline) vs {eo['abs_rel']:.3f} (ours), "
              f"RMSE {eb['rmse']:.2f} vs {eo['rmse']:.2f}")
    print("\n-> Done!")


if __name__ == "__main__":
    lt.monkey_patch()
    conf = Conf().conf
    main(conf)
