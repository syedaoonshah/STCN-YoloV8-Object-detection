"""
STCN Video Segmentation with Object Detection & Labeling
========================================================
Detect specific objects and label them on the output
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import hashlib

# Dynamic color generation function
def get_color_for_object(obj_name):
    """Generate a unique BGR color for an object name using hash"""
    # Create a hash of the object name
    hash_obj = hashlib.md5(obj_name.encode())
    hash_bytes = hash_obj.digest()
    # Use first 3 bytes as BGR values
    b = int(hash_bytes[0]) % 256
    g = int(hash_bytes[1]) % 256
    r = int(hash_bytes[2]) % 256
    # Ensure minimum brightness for visibility
    if b + g + r < 100:
        b = (b + 150) % 256
        g = (g + 150) % 256
        r = (r + 150) % 256
    return (b, g, r)

# YOLOv8 object detection (lightweight)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("YOLOv8 not installed, installing now...")
    os.system("pip install -q ultralytics")
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except:
        YOLO_AVAILABLE = False


class STCNObjectSegmenter:
    """STCN with object detection and labeling"""
    
    def __init__(self, device='cuda', target_objects=None):
        self.device = torch.device(device)
        self.target_objects = [obj.lower() for obj in (target_objects or [])]  # Empty = detect all
        self.color_cache = {}  # Cache colors for consistent object coloring
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print(f"Using device: {self.device}")
        if self.target_objects:
            print(f"Filtering objects: {', '.join(self.target_objects)}")
        else:
            print("Detecting all COCO objects")
        
        self._load_detector()
    
    def _load_detector(self):
        """Load YOLOv8 for object detection"""
        global YOLO_AVAILABLE
        
        if not YOLO_AVAILABLE:
            print("⚠ YOLOv8 not available, installing...")
            os.system("pip install -q ultralytics")
            try:
                from ultralytics import YOLO
                YOLO_AVAILABLE = True
            except Exception as e:
                print(f"Failed to install YOLOv8: {e}")
                self.detector = None
                return
        
        print("\nLoading YOLOv8 for object detection...")
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')  # nano model for speed
            self.detector.to(self.device)
            print("✓ YOLOv8 loaded")
        except Exception as e:
            print(f"⚠ YOLOv8 load failed: {e}")
            self.detector = None
    
    def detect_objects(self, frame_rgb):
        """Detect objects in frame using YOLOv8"""
        if self.detector is None:
            return {}
        
        try:
            results = self.detector(frame_rgb, verbose=False)[0]
            detections = {}
            
            for box in results.boxes:
                class_id = int(box.cls)
                class_name = results.names[class_id].lower()
                conf = float(box.conf)
                
                # Filter by target objects if specified, otherwise detect all
                if conf > 0.3:
                    if self.target_objects and class_name not in self.target_objects:
                        continue  # Skip if not in target list
                    
                    if class_name not in detections:
                        detections[class_name] = []
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections[class_name].append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'centroid': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
            
            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return {}
    
    def create_mask_from_detections(self, frame, detections):
        """Create segmentation mask from object detections"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        obj_id = 1
        for obj_name, detections_list in detections.items():
            for det in detections_list:
                x1, y1, x2, y2 = det['bbox']
                # Create mask region for detected object
                mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = obj_id
                obj_id += 1
        
        return mask, detections
    
    def create_overlay_with_labels(self, frame, detections):
        """Create overlay with labels at object centroids - no bounding boxes"""
        overlay = frame.copy()
        
        for obj_name, detections_list in detections.items():
            # Get or generate color for this object
            if obj_name not in self.color_cache:
                self.color_cache[obj_name] = get_color_for_object(obj_name)
            color = self.color_cache[obj_name]
            
            for det in detections_list:
                bbox = det['bbox']
                conf = det['conf']
                
                # Apply color overlay only within bounding box region
                x1, y1, x2, y2 = bbox
                # Blend the region: 70% original + 30% color
                region = overlay[y1:y2, x1:x2]
                color_overlay = np.full_like(region, color)
                overlay[y1:y2, x1:x2] = cv2.addWeighted(region, 0.7, color_overlay, 0.3, 0)
                
                # Add label at centroid
                cx, cy = det['centroid']
                label = f"{obj_name.upper()}"
                
                # Get text size for centering
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                
                # Draw text with white background for visibility
                cv2.putText(overlay, label, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        return overlay


def process_video_with_detection(frame_dir, output_dir, target_objects):
    """Process video with object detection and labeling"""
    print(f"\n{'='*60}")
    print(f"Processing: {frame_dir}")
    print(f"Target objects: {', '.join(target_objects)}")
    print(f"{'='*60}")
    
    # Get frames
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    
    if not frame_files:
        print("No frames found!")
        return
    
    # Load first frame
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    height, width = first_frame.shape[:2]
    
    print(f"Frames: {width}x{height}, {len(frame_files)} frames")
    
    # Initialize segmenter
    segmenter = STCNObjectSegmenter(device='cuda', target_objects=target_objects)
    
    # Create output
    os.makedirs(output_dir, exist_ok=True)
    
    # Video writers
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    overlay_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'stcn_labeled_overlay.mp4'),
        fourcc, fps, (width, height)
    )
    
    mask_writer = cv2.VideoWriter(
        os.path.join(output_dir, 'stcn_labeled_mask.mp4'),
        fourcc, fps, (width, height)
    )
    
    print("\nProcessing frames with object detection...")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Detecting & Segmenting")):
        frame = cv2.imread(os.path.join(frame_dir, frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect objects
        detections = segmenter.detect_objects(frame_rgb)
        
        # Create visualizations
        overlay = segmenter.create_overlay_with_labels(frame, detections)
        
        # Create mask
        mask, _ = segmenter.create_mask_from_detections(frame, detections)
        colored_mask = np.zeros_like(frame)
        
        for obj_name, detections_list in detections.items():
            if obj_name not in segmenter.color_cache:
                segmenter.color_cache[obj_name] = get_color_for_object(obj_name)
            color = segmenter.color_cache[obj_name]
            for det in detections_list:
                x1, y1, x2, y2 = det['bbox']
                colored_mask[y1:y2, x1:x2] = color
        
        # Write frames
        overlay_writer.write(overlay)
        mask_writer.write(colored_mask)
    
    overlay_writer.release()
    mask_writer.release()
    
    print(f"\n✓ Output saved to: {output_dir}")
    print(f"  - stcn_labeled_overlay.mp4")
    print(f"  - stcn_labeled_mask.mp4")


def main():
    print("="*60)
    print("STCN OBJECT DETECTION & SEGMENTATION")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage: python stcn_object_segmentation.py <frame_dir> [output_dir] [object1,object2,...]")
        print("\nExamples:")
        print("  # Detect all COCO objects:")
        print("  python stcn_object_segmentation.py dataset/horse/horses-kids output/stcn")
        print("\n  # Filter specific objects:")
        print("  python stcn_object_segmentation.py dataset/horse/horses-kids output/stcn person,horse")
        print("  python stcn_object_segmentation.py dataset/traffic output/stcn car,bus,truck")
        sys.exit(1)
    
    frame_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/stcn_detection"
    
    # Parse target objects (empty list = detect all)
    target_objects = []
    if len(sys.argv) > 3:
        target_objects = [obj.strip().lower() for obj in sys.argv[3].split(',')]
    
    if not os.path.exists(frame_dir):
        print(f"Error: Directory not found: {frame_dir}")
        sys.exit(1)
    
    process_video_with_detection(frame_dir, output_dir, target_objects)
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
