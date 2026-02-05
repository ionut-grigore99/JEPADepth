'''Script to clean up Cityscapes test data by removing files not in cityscapes_test_files.txt
This helps save disk space by keeping only the necessary test files for evaluation.
'''

import argparse
import os
from glob import glob
from tqdm import tqdm

def load_required_files(test_file_list):
    """Load the list of required frame IDs and their temporal neighbors from cityscapes_test_files.txt"""
    required_frames = set()
    
    print(f"Loading required files from {test_file_list}...")
    with open(test_file_list, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                city, frame_id = parts
                
                # Parse frame components
                # frame_id format: city_sequence_framenum (e.g., berlin_000000_000019)
                city_name, seq, frame_num = frame_id.split('_')
                frame_num = int(frame_num)
                
                # Add the main frame and its temporal neighbors (offset -2, 0, +2 for seq_length=3)
                for offset in [-2, 0, 2]:
                    neighbor_num = str(frame_num + offset).zfill(6)
                    neighbor_id = f"{city_name}_{seq}_{neighbor_num}_"
                    required_frames.add((city_name, neighbor_id))
    
    print(f"Identified {len(required_frames)} required frames (including temporal neighbors)")
    return required_frames

def cleanup_images(dataset_dir, required_frames, dry_run=False):
    """Remove unnecessary image files from leftImg8bit_sequence/test/"""
    img_dir = os.path.join(dataset_dir, 'leftImg8bit_sequence', 'test')
    
    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} does not exist!")
        return
    
    deleted_count = 0
    kept_count = 0
    total_size_deleted = 0
    
    print("\nProcessing image files...")
    city_list = os.listdir(img_dir)
    
    for city in tqdm(city_list, desc="Cities"):
        city_path = os.path.join(img_dir, city)
        if not os.path.isdir(city_path):
            continue
        
        img_files = glob(os.path.join(city_path, '*.png'))
        
        for img_file in img_files:
            frame_id = os.path.basename(img_file).split('leftImg8bit')[0]
            
            # Check if this frame is required
            if (city, frame_id) not in required_frames:
                file_size = os.path.getsize(img_file)
                total_size_deleted += file_size
                
                if not dry_run:
                    os.remove(img_file)
                deleted_count += 1
            else:
                kept_count += 1
    
    size_mb = total_size_deleted / (1024 * 1024)
    size_gb = size_mb / 1024
    
    if dry_run:
        print(f"\n[DRY RUN] Would delete {deleted_count} image files (~{size_gb:.2f} GB)")
    else:
        print(f"\nDeleted {deleted_count} image files (~{size_gb:.2f} GB)")
    print(f"Kept {kept_count} required image files")

def cleanup_cameras(dataset_dir, required_frames, dry_run=False):
    """Remove unnecessary camera files from camera_trainvaltest/camera/test/"""
    camera_dir = os.path.join(dataset_dir, 'camera_trainvaltest', 'camera', 'test')
    
    if not os.path.exists(camera_dir):
        # Try alternative path
        camera_dir = os.path.join(dataset_dir, 'camera', 'test')
        if not os.path.exists(camera_dir):
            print(f"Warning: Camera directory does not exist!")
            return
    
    deleted_count = 0
    kept_count = 0
    total_size_deleted = 0
    
    print("\nProcessing camera files...")
    city_list = os.listdir(camera_dir)
    
    for city in tqdm(city_list, desc="Cities"):
        city_path = os.path.join(camera_dir, city)
        if not os.path.isdir(city_path):
            continue
        
        camera_files = glob(os.path.join(city_path, '*_camera.json'))
        
        for camera_file in camera_files:
            # Extract frame_id from camera file name
            # Format: city_sequence_framenum_camera.json
            basename = os.path.basename(camera_file)
            frame_id = basename.replace('_camera.json', '') + '_'
            
            # Check if this frame is required
            if (city, frame_id) not in required_frames:
                file_size = os.path.getsize(camera_file)
                total_size_deleted += file_size
                
                if not dry_run:
                    os.remove(camera_file)
                deleted_count += 1
            else:
                kept_count += 1
    
    size_kb = total_size_deleted / 1024
    
    if dry_run:
        print(f"\n[DRY RUN] Would delete {deleted_count} camera files (~{size_kb:.2f} KB)")
    else:
        print(f"\nDeleted {deleted_count} camera files (~{size_kb:.2f} KB)")
    print(f"Kept {kept_count} required camera files")

def main():
    parser = argparse.ArgumentParser(description='Clean up unnecessary Cityscapes test files')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to Cityscapes dataset (e.g., data/cityscapes)')
    parser.add_argument('--test_file_list', type=str, required=True,
                        help='Path to cityscapes_test_files.txt')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')
    args = parser.parse_args()
    
    if not os.path.exists(args.test_file_list):
        print(f"Error: Test file list not found: {args.test_file_list}")
        return
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    
    print("="*80)
    print("CITYSCAPES TEST DATA CLEANUP")
    print("="*80)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Test file list: {args.test_file_list}")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'CLEANUP (files will be deleted)'}")
    print("="*80)
    
    # Load required files
    required_frames = load_required_files(args.test_file_list)
    
    # Cleanup images
    cleanup_images(args.dataset_dir, required_frames, args.dry_run)
    
    # Cleanup camera files
    cleanup_cameras(args.dataset_dir, required_frames, args.dry_run)
    
    print("\n" + "="*80)
    if args.dry_run:
        print("DRY RUN COMPLETE - No files were deleted")
        print("Run without --dry-run to actually delete files")
    else:
        print("CLEANUP COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
