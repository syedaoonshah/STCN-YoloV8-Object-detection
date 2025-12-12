"""
SAM Video Segmentation with Custom Prompts
===========================================
Uses Meta's Segment Anything Model with point/box/text prompts
Detects ANY object you specify with prompts
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import requests
from tqdm import tqdm


class SAMVideoSegmenter:
    """Video segmentation using SAM with prompts"""
    
    def __init__(self, device='cuda', force_gpu=True):
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        
        if device == 'cuda' and not gpu_available:
            if force_gpu:
                print("ERROR: CUDA not available. GPU required for this operation.")
                print("Please ensure NVIDIA GPU drivers and CUDA toolkit are installed.")
                sys.exit(1)
            else:
                print("WARNING: CUDA not available. Falling back to CPU (slower).")
                device = 'cpu'
        
        if gpu_available:
            print(f"GPU Found: {torch.cuda.get_device_name(0)}")
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.checkpoint_path = self.download_checkpoint()
        self.load_sam()
    
    def download_checkpoint(self):
        """Download SAM checkpoint if needed"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"Using existing SAM checkpoint: {checkpoint_path}")
            return checkpoint_path
        
        print("Downloading SAM checkpoint (2.4GB)...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total = int(response.headers.get('content-length', 0))
            with open(checkpoint_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print("✓ Download complete")
            return checkpoint_path
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            print(f"Please download manually from: {url}")
            sys.exit(1)
    
    def load_sam(self):
        """Load SAM model"""
        print("\nLoading SAM model...")
        sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("✓ SAM loaded successfully")
    
    def segment_with_points(self, image, points, labels):
        """
        Segment image with point prompts
        points: np.array of [x, y] coordinates
        labels: np.array of 1 (foreground) or 0 (background)
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx], logits[best_idx]
    
    def segment_with_box(self, image, box):
        """
        Segment image with bounding box
        box: [x1, y1, x2, y2]
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx], logits[best_idx]
    
    def segment_everything(self, image):
        """Segment everything in the image automatically"""
        from segment_anything import SamAutomaticMaskGenerator
        
        mask_generator = SamAutomaticMaskGenerator(
            sam_model_registry["vit_h"](checkpoint=self.checkpoint_path).to(self.device)
        )
        
        masks = mask_generator.generate(image)
        return masks
    
    def create_colored_mask(self, mask, color=(255, 0, 0)):
        """Create colored visualization of mask"""
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[mask] = color
        return colored
    
    def create_overlay(self, image, mask, alpha=0.5, color=(255, 0, 0)):
        """Create overlay visualization"""
        colored_mask = self.create_colored_mask(mask, color)
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlay


def get_prompt_points_interactive(frame, num_points=3):
    """Interactively select points on the first frame"""
    print("\n" + "="*60)
    print("INTERACTIVE PROMPT MODE")
    print("="*60)
    print("Instructions:")
    print("  - Click on the objects you want to segment")
    print("  - Press SPACE to finish")
    print("  - Press ESC to cancel")
    print("="*60)
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            # Draw point
            cv2.circle(frame_display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame_display, f"{len(points)}", (x+15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Select Points', frame_display)
    
    frame_display = frame.copy()
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', mouse_callback)
    cv2.imshow('Select Points', frame_display)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    
    if not points:
        return None
    
    return np.array(points)


def process_video_with_sam(frame_dir, output_dir, target_objects=None, prompt_mode='auto'):
    """
    Process video with SAM
    target_objects: list of objects to detect (e.g., ['person', 'horse', 'animal'])
    prompt_mode: 'auto', 'points', or 'interactive'
    """
    print(f"\n{'='*60}")
    print(f"Processing: {frame_dir}")
    print(f"Target objects: {target_objects if target_objects else 'ALL'}")
    print(f"Prompt mode: {prompt_mode}")
    print(f"{'='*60}")
    
    # Object hints for better detection
    object_hints = {
        'person': 'A human figure or person',
        'horse': 'A horse or equine animal',
        'animal': 'Any animal',
        'dog': 'A dog or canine',
        'cat': 'A cat or feline',
        'car': 'A car or vehicle',
        'bird': 'A bird',
        'cow': 'A cow or bovine'
    }
    
    if target_objects:
        print(f"\nTarget object descriptions:")
        for obj in target_objects:
            if obj.lower() in object_hints:
                print(f"  - {obj}: {object_hints[obj.lower()]}")
            else:
                print(f"  - {obj}")
    
    # Get frames
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    
    if not frame_files:
        print("No frames found!")
        return
    
    # Load first frame
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    height, width = first_frame.shape[:2]
    
    # Downscale for GPU memory (RTX 4060 has 8GB)
    # SAM needs significant VRAM for high-res images
    MAX_SIZE = 720  # Reduced to 720p for safer processing
    scale = 1.0
    if max(height, width) > MAX_SIZE:
        scale = MAX_SIZE / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        first_frame_rgb = cv2.resize(first_frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        height, width = new_height, new_width
        print(f"Downscaled frames: {int(first_frame.shape[1])}x{int(first_frame.shape[0])} -> {width}x{height}")
    else:
        print(f"Frame resolution: {width}x{height}")
    
    print(f"Frames: {width}x{height}, {len(frame_files)} frames")
    
    # Initialize SAM with GPU
    segmenter = SAMVideoSegmenter(device='cuda', force_gpu=False)
    
    # Get prompts
    if prompt_mode == 'interactive':
        prompt_points = get_prompt_points_interactive(first_frame.copy())
        if prompt_points is None:
            print("No points selected. Exiting.")
            return
    elif prompt_mode == 'auto':
        # Auto points: center and some distributed points
        prompt_points = np.array([
            [width // 2, height // 2],
            [width // 3, height // 2],
            [2 * width // 3, height // 2],
        ])
    elif prompt_points is None:
        # Default center point
        prompt_points = np.array([[width // 2, height // 2]])
    
    print(f"\nUsing {len(prompt_points)} prompt points:")
    for i, pt in enumerate(prompt_points):
        print(f"  Point {i+1}: ({pt[0]}, {pt[1]})")
    
    # Segment first frame to get initial mask
    print("\nSegmenting first frame...")
    labels = np.array([1] * len(prompt_points))  # All foreground
    mask, score, logit = segmenter.segment_with_points(first_frame_rgb, prompt_points, labels)
    
    coverage = mask.sum() / (mask.shape[0] * mask.shape[1]) * 100
    print(f"✓ First frame segmented (coverage: {coverage:.1f}%, confidence: {score:.3f})")
    
    # Create output
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writers
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'sam_overlay.mp4'),
        fourcc, fps, (width, height)
    )
    mask_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'sam_mask.mp4'),
        fourcc, fps, (width, height)
    )
    comparison_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'sam_comparison.mp4'),
        fourcc, fps, (width * 2, height)
    )
    
    # Process all frames
    print("\nSegmenting all frames...")
    current_logit = logit
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing")):
        frame = cv2.imread(os.path.join(frame_dir, frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply same scaling
        if scale < 1.0:
            new_w = int(frame_rgb.shape[1] * scale)
            new_h = int(frame_rgb.shape[0] * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if idx == 0:
            current_mask = mask
        else:
            # Refine with previous mask
            segmenter.predictor.set_image(frame_rgb)
            masks, scores, logits = segmenter.predictor.predict(
                mask_input=current_logit[None, :, :],
                multimask_output=True
            )
            best_idx = np.argmax(scores)
            current_mask = masks[best_idx]
            current_logit = logits[best_idx]
        
        # Create visualizations
        overlay = segmenter.create_overlay(frame, current_mask, alpha=0.5, color=(0, 255, 0))
        colored_mask = segmenter.create_colored_mask(current_mask, color=(0, 255, 0))
        comparison = np.hstack([frame, overlay])
        
        # Add text label
        cv2.putText(overlay, 'SAM SEGMENTATION', (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Write
        overlay_writer.write(overlay)
        mask_writer.write(colored_mask)
        comparison_writer.write(comparison)
    
    overlay_writer.release()
    mask_writer.release()
    comparison_writer.release()
    
    print(f"\n✓ Output saved to: {output_dir}")
    print("  - sam_overlay.mp4")
    print("  - sam_mask.mp4")
    print("  - sam_comparison.mp4")


def main():
    print("="*60)
    print("SAM VIDEO SEGMENTATION WITH CUSTOM PROMPTS")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python sam_segmentation.py <frame_dir> [output_dir] [objects] [mode]")
        print("\nObjects (comma-separated):")
        print("  person, horse, animal, dog, cat, car, bird, cow")
        print("\nModes:")
        print("  interactive - Click points on first frame (default)")
        print("  auto        - Auto-select center points")
        print("\nExamples:")
        print("  python sam_segmentation.py dataset/horse/horses-kids output/test person,horse auto")
        print("  python sam_segmentation.py dataset/horse/horses-kids output/test horse interactive")
        print("  python sam_segmentation.py dataset/horse/horses-kids output/test animal auto")
        sys.exit(1)
    
    frame_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/sam_output"
    
    # Parse target objects and mode
    target_objects = None
    prompt_mode = 'auto'
    
    if len(sys.argv) > 3:
        # Check if it's objects or mode
        arg3 = sys.argv[3]
        if arg3 in ['interactive', 'auto']:
            prompt_mode = arg3
        else:
            # Parse as objects
            target_objects = [obj.strip().lower() for obj in arg3.split(',')]
            
            # Check for mode in argument 4
            if len(sys.argv) > 4:
                if sys.argv[4] in ['interactive', 'auto']:
                    prompt_mode = sys.argv[4]
    
    if not os.path.exists(frame_dir):
        print(f"Error: Directory not found: {frame_dir}")
        sys.exit(1)
    
    process_video_with_sam(frame_dir, output_dir, target_objects, prompt_mode)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
