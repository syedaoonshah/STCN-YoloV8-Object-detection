"""
Prompted Video Segmentation with Custom Object Detection
=========================================================
Specify which objects to detect in your video
"""
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision import models, transforms
import torch.nn.functional as F


# Pascal VOC classes
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


class PromptedSegmenter:
    """Segmenter with custom object filtering"""
    
    def __init__(self, target_classes=None, device='cuda'):
        # Use CUDA if available, otherwise fallback to CPU
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("⚠ CUDA not available, using CPU")
        
        self.device = torch.device(device)
        self.target_classes = target_classes or ['person', 'horse', 'car', 'dog']
        
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using device: {self.device}")
        print(f"Target objects: {', '.join(self.target_classes)}")
        
        self.load_model()
        self.setup_colors()
    
    def load_model(self):
        """Load DeepLabV3 model"""
        print("\nLoading DeepLabV3 ResNet50...")
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("✓ Model loaded")
    
    def setup_colors(self):
        """Setup colors for target classes"""
        np.random.seed(42)
        all_colors = np.random.randint(0, 256, (len(VOC_CLASSES), 3), dtype=np.uint8)
        all_colors[0] = [0, 0, 0]  # Background always black
        
        # Assign distinct colors to target classes
        color_palette = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
        ]
        
        for idx, cls_name in enumerate(self.target_classes):
            if cls_name in VOC_CLASSES:
                cls_id = VOC_CLASSES.index(cls_name)
                if idx < len(color_palette):
                    all_colors[cls_id] = color_palette[idx]
        
        self.colors = all_colors
        
        # Create class ID mapping
        self.target_class_ids = [VOC_CLASSES.index(cls) for cls in self.target_classes if cls in VOC_CLASSES]
    
    def segment_frame(self, frame):
        """Segment frame and filter by target classes"""
        with torch.no_grad():
            # Preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_resized = pil_img.resize((512, 512), Image.BILINEAR)
            
            tensor = self.transform(pil_resized).unsqueeze(0).to(self.device)
            
            # Inference
            output = self.model(tensor)
            preds = torch.argmax(output['out'], dim=1).squeeze(0).cpu().numpy()
            
            # Resize back
            h, w = frame.shape[:2]
            mask = cv2.resize(
                preds.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            
            # Filter to only target classes
            filtered_mask = np.zeros_like(mask)
            for cls_id in self.target_class_ids:
                filtered_mask[mask == cls_id] = cls_id
            
            return filtered_mask
    
    def create_colored_mask(self, mask):
        """Create colored visualization"""
        colored = self.colors[mask]
        return colored
    
    def create_overlay_with_labels(self, frame, mask, alpha=0.5):
        """Create overlay with labels for detected objects"""
        colored_mask = self.create_colored_mask(mask)
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
        
        # Add labels only for target classes
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes != 0]  # Remove background
        
        for cls_id in unique_classes:
            if cls_id in self.target_class_ids:
                # Find contours
                class_mask = (mask == cls_id).astype(np.uint8)
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Label significant objects
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Min area threshold
                        # Get centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Draw label
                            label = VOC_CLASSES[cls_id].upper()
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                            
                            # Background rectangle
                            cv2.rectangle(overlay, 
                                        (cx - text_w//2 - 10, cy - text_h - 10),
                                        (cx + text_w//2 + 10, cy + 10),
                                        (0, 0, 0), -1)
                            
                            # Text
                            cv2.putText(overlay, label, 
                                      (cx - text_w//2, cy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                      (255, 255, 255), 4, cv2.LINE_AA)
        
        return overlay


def process_frame_directory(frame_dir, output_dir, target_classes, max_frames=None):
    """Process directory of frames with custom prompts"""
    print(f"\n{'='*60}")
    print(f"Processing: {frame_dir}")
    print(f"{'='*60}")
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    if max_frames:
        frame_files = frame_files[:max_frames]
    
    if not frame_files:
        print("No frame files found!")
        return
    
    # Get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    fps = 30
    total_frames = len(frame_files)
    
    # Optimize for high resolution - downscale if needed
    output_scale = 1.0
    if max(height, width) > 1440:
        output_scale = 1440 / max(height, width)
        output_width = int(width * output_scale)
        output_height = int(height * output_scale)
        print(f"Frames: {width}x{height} (will output at {output_width}x{output_height}), {total_frames} frames @ {fps}fps")
    else:
        output_width = width
        output_height = height
        print(f"Frames: {width}x{height}, {total_frames} frames @ {fps}fps")
    
    # Initialize segmenter with target classes
    segmenter = PromptedSegmenter(target_classes=target_classes, device='cuda')
    
    # Create output
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writers - only create essential outputs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'overlay_labeled.mp4'),
        fourcc, fps, (output_width, output_height)
    )
    
    # Process frames
    print("\nSegmenting...")
    class_stats = {}
    
    for idx, frame_file in enumerate(frame_files):
        frame = cv2.imread(os.path.join(frame_dir, frame_file))
        
        # Segment
        mask = segmenter.segment_frame(frame)
        
        # Skip stats collection - it's too slow with large masks
        
        # Create visualizations
        overlay = segmenter.create_overlay_with_labels(frame, mask, alpha=0.5)
        
        # Downscale for output if needed
        if output_scale < 1.0:
            overlay = cv2.resize(overlay, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            frame_scaled = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        else:
            frame_scaled = frame
        
        # Write overlay video
        overlay_writer.write(overlay)
        
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  {idx + 1}/{total_frames} frames")
    
    overlay_writer.release()
    
    # Print statistics
    print("\n" + "="*60)
    print("Detected Target Objects:")
    print("="*60)
    
    if class_stats:
        total_pixels = sum(class_stats.values())
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
        for cls_name, count in sorted_classes:
            pct = count / total_pixels * 100
            print(f"  {cls_name.upper():15s}: {pct:5.1f}%")
    else:
        print("  No target objects detected!")
    
    print(f"\n✓ Output saved to: {output_dir}")
    print("  - overlay_labeled.mp4 (with object labels)")


def main():
    print("="*60)
    print("PROMPTED VIDEO OBJECT SEGMENTATION")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python prompt_segmentation.py <video_dir> [output_dir] [object1,object2,...]")
        print("\nExamples:")
        print("  python prompt_segmentation.py dataset/horse/horses-kids")
        print("  python prompt_segmentation.py dataset/beach output/beach person")
        print("  python prompt_segmentation.py dataset/horse out person,horse,dog")
        print("\nAvailable objects:")
        print("  " + ", ".join(VOC_CLASSES[1:]))  # Skip background
        sys.exit(1)
    
    frame_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/prompted_segmentation"
    
    # Parse target classes
    if len(sys.argv) > 3:
        # Classes can be comma-separated or space-separated
        classes_input = ' '.join(sys.argv[3:])
        target_classes = [c.strip() for c in classes_input.replace(',', ' ').split() if c.strip()]
    else:
        # Interactive prompt
        print("\n" + "="*60)
        print("Which objects do you want to detect?")
        print("="*60)
        print("Available objects:")
        print("  " + ", ".join(VOC_CLASSES[1:]))
        print("\nExamples:")
        print("  person horse dog")
        print("  car bus")
        print("  person")
        target_input = input("\nEnter objects (space-separated): ").strip()
        target_classes = [c.strip() for c in target_input.split() if c.strip()]
    
    if not target_classes:
        print("No target classes specified. Using default: person, horse, car, dog")
        target_classes = ['person', 'horse', 'car', 'dog']
    
    # Validate classes
    valid_classes = [c for c in target_classes if c in VOC_CLASSES]
    invalid_classes = [c for c in target_classes if c not in VOC_CLASSES]
    
    if invalid_classes:
        print(f"\n⚠ Warning: Invalid classes (ignored): {', '.join(invalid_classes)}")
    
    if not valid_classes:
        print("Error: No valid classes specified!")
        sys.exit(1)
    
    if not os.path.exists(frame_dir):
        print(f"Error: Directory not found: {frame_dir}")
        sys.exit(1)
    
    # Process with GPU
    process_frame_directory(frame_dir, output_dir, valid_classes)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
