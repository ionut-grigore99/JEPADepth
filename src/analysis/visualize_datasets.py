"""
Visualization script for KITTI dataset samples.
This script loads and visualizes samples from different KITTI splits to inspect and analyze the data structure, quality, and characteristics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import random as rand

from src.datasets.kitti_dataset import KITTIRAWDataset
from src.utils import readlines


def visualize_sample(dataset, sample_idx, split_name, save_path=None):
    """
    Visualize a single sample from the dataset in a clean, well-structured layout.
    """
    sample = dataset[sample_idx]

    # Better layout handling
    fig = plt.figure(figsize=(20, 12), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1])

    fig.suptitle(f'KITTI Dataset Sample - {split_name} split (Index: {sample_idx})', fontsize=18, fontweight='bold', y=1.02)

    # === Row 1: Input frames ===
    frame_ids = [0, -1, 1]
    titles = ['Target Frame (t)', 'Previous Frame (t-1)', 'Next Frame (t+1)']

    for idx, (frame_id, title) in enumerate(zip(frame_ids, titles)):
        ax = fig.add_subplot(gs[0, idx])
        img = sample[("color", frame_id, 0)].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis('off')

    # === Row 1, Col 4: Dataset info and camera intrinsics ===
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')

    # Camera intrinsics
    K = sample[("K", 0)].cpu().numpy()
    K_str = "\n".join("  ".join(f"{v:8.2f}" for v in row) for row in K)

    # Build combined info block
    info_text = (
        "Image Resolution (used for this visualization):\n"
        f"  Height: {dataset.height}\n"
        f"  Width:  {dataset.width}\n\n"
        "Camera Intrinsics (K):\n"
        f"{K_str}"
    )

    # Draw text box
    ax.text(
        0.05, 0.5,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )


    # === Row 2: Depth visualizations ===
    if "depth_gt" in sample:
        depth_gt = sample["depth_gt"].squeeze().cpu().numpy()
        valid_mask = depth_gt > 0
        valid_depths = depth_gt[valid_mask]

        # --- Col 0: Depth map ---
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(depth_gt, cmap='viridis', vmin=0, vmax=80, interpolation='nearest')
        ax.set_title(f"Ground Truth Depth\nValid pixels: {100*valid_mask.mean():.1f}%", fontsize=11, pad=6)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Col 1: Histogram ---
        ax = fig.add_subplot(gs[1, 1])
        ax.hist(valid_depths, bins=50, color='steelblue', alpha=0.8, edgecolor='black')
        ax.set_title("Depth Distribution", fontsize=11, pad=6)
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Pixel count")
        ax.grid(alpha=0.3)

        # --- Col 2: Garg/Eigen crop overlay ---
        ax = fig.add_subplot(gs[1, 2])
        depth_rgb = plt.cm.viridis(depth_gt / 80.0)[..., :3]
        overlay = depth_rgb.copy()

        # crop dims
        top, bottom = 153, 371
        left, right = 44, 1197

        overlay[:top] *= 0.3
        overlay[bottom:] *= 0.3
        overlay[:, :left] *= 0.3
        overlay[:, right:] *= 0.3

        ax.imshow(overlay)
        ax.set_title("Garg/Eigen Crop Region \nUsed for validation metrics only (no gradient backpropagation)", fontsize=10, pad=6)
        ax.axis('off')

        from matplotlib.patches import Rectangle
        rect = Rectangle((left, top), right-left, bottom-top, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect)

        # --- Col 3: Depth range stats ---
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')

        ranges = [(0, 10), (10, 20), (20, 40), (40, 80)]
        range_stats = []
        for r_min, r_max in ranges:
            mask = (valid_depths >= r_min) & (valid_depths < r_max)
            pct = 100 * mask.sum() / len(valid_depths)
            range_stats.append(f"{r_min:2d}-{r_max:2d}m: {pct:5.1f}% (out of valid pixels)")

        stats_text = (
            "Depth Range Statistics:\n"
            "----------------------\n" +
            "\n".join(range_stats) +
            f"\nTotal valid pixels: {100*valid_mask.mean():.2f}%"
        )

        ax.text(0.05, 0.6, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', family='monospace', linespacing=1.0, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # Place arrow symbol midway between the two axes centers
        # 1) Target -> Previous (put symbol between them)
        fig.text(0.26, 0.84, "→", ha="center", va="center", fontsize=70, fontweight="bold")

        # 2) Target -> Depth (put symbol between them)
        fig.text(0.12, 0.68, "↓", ha="center", va="center", fontsize=70, fontweight="bold")

    else:
        ax = fig.add_subplot(gs[2, :])
        ax.text(0.5, 0.5, "No ground truth depth available", ha='center', va='center', fontsize=14, color='red')
        ax.axis('off')
        

    # Save
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig



def compare_splits(data_path, output_dir):
    """
    Compare samples from different KITTI splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        'eigen_zhou': ('train_files.txt', True),
        'eigen_full': ('train_files.txt', True),
        'eigen': ('test_files.txt', False),
        'eigen_benchmark': ('test_files.txt', False),
        'benchmark': ('train_files.txt', True),
    }
    
    for split_name, (filename, is_train) in splits.items():
        print(f"\nProcessing split: {split_name}")
        
        split_path = Path(data_path).parent / "kitti_splits" / split_name / filename
        
        if not split_path.exists():
            print(f"Split file not found: {split_path}")
            print(f"Skipping {split_name}...")
            continue
        
        # Load filenames
        filenames = readlines(split_path)
        print(f"Found {len(filenames)} samples in {split_name}")
        
        # Create dataset
        try:
            dataset = KITTIRAWDataset(
                data_path=data_path,
                filenames=filenames,
                height=192,
                width=640,
                frame_idxs=[0, -1, 1],
                num_scales=1,
                is_train=is_train,
                img_ext='.jpg'
            )

            # Visualize a random sample
            sample_idx = rand.randint(0, len(dataset) - 1)
            fig = visualize_sample(dataset=dataset, sample_idx=sample_idx, split_name=split_name, save_path=os.path.join(output_dir, f'{split_name}_sample.png'))
            plt.close(fig)
        
            
        except Exception as e:
            print(f"Error processing {split_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll visualizations saved to: {output_dir}/")


def analyze_split_statistics(data_path):
    """
    Analyze and print statistics for all KITTI splits.
    """
    print("KITTI Split Statistics:")
    
    splits_info = {
        'eigen_zhou': {
            'train': 'train_files.txt',
            'val': 'val_files.txt',
        },
        'eigen_full': {
            'train': 'train_files.txt',
            'val': 'val_files.txt',
        },
        'eigen': {
            'test': 'test_files.txt',
        },
        'eigen_benchmark': {
            'test': 'test_files.txt',
        },
        'benchmark': {
            'train': 'train_files.txt',
            'val': 'val_files.txt',
        },
    }
    
    split_base_path = Path(data_path).parent / "kitti_splits"
    
    results = []
    
    for split_name, files in splits_info.items():
        split_path = split_base_path / split_name
        
        if not split_path.exists():
            continue
        
        split_stats = {'name': split_name}
        
        for subset_name, filename in files.items():
            if filename is None:
                continue
                
            file_path = split_path / filename
            if file_path.exists():
                filenames = readlines(file_path)
                split_stats[subset_name] = len(filenames)
            else:
                split_stats[subset_name] = 0
        
        results.append(split_stats)
    
    # Print table
    print(f"{'Split Name':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)
    
    for stats in results:
        name = stats['name']
        train = stats.get('train', 0)
        val = stats.get('val', 0)
        test = stats.get('test', 0)
        total = train + val + test
        
        print(f"{name:<20} {train:<10} {val:<10} {test:<10} {total:<10}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'data/kitti/kitti_data'
    OUTPUT_DIR = 'assets/dataset_visualizations'
    
    print("\n" + "=" * 80)
    print("KITTI Dataset Inspector")
    print("=" * 80)
    print(f"\nData path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Check if data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH in the script to point to your KITTI data directory.")
        sys.exit(1)
    
    # Analyze split statistics
    analyze_split_statistics(DATA_PATH)
    
    # Visualize samples from each split
    compare_splits(DATA_PATH, OUTPUT_DIR)
    
    print("Dataset inspection complete!")
    print("=" * 80 + "\n")
