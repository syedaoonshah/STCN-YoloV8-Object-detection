"""
Convert MP4 videos to GIFs
"""
import os
import cv2
import imageio
from pathlib import Path
from tqdm import tqdm

def mp4_to_gif(mp4_path, gif_path, fps=5, scale=1.0):
    """Convert MP4 video to GIF with optional scaling"""
    print(f"\nConverting: {mp4_path}")
    
    cap = cv2.VideoCapture(mp4_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {mp4_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scale dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f"Resolution: {width}x{height} -> {new_width}x{new_height}")
    print(f"Frames: {frame_count}, Original FPS: {original_fps}")
    
    frames = []
    frame_idx = 0
    frame_step = max(1, int(original_fps / fps))
    
    print(f"Reading frames (every {frame_step}th frame for {fps} fps output)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Read every Nth frame to achieve desired fps
        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if scaling
            if scale != 1.0:
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    
    print(f"Saving GIF with {len(frames)} frames...")
    # Duration in milliseconds per frame (lower fps = higher duration = slower playback)
    duration = 1000 / fps
    imageio.mimsave(gif_path, frames, duration=duration, loop=0)
    print(f"✓ GIF saved: {gif_path} (duration: {duration:.0f}ms per frame)")

def process_output_folder(output_dir, fps=5, scale=0.5):
    """Process all MP4 files in output directory and subdirectories"""
    output_path = Path(output_dir)
    
    mp4_files = list(output_path.rglob('*.mp4'))
    
    if not mp4_files:
        print(f"No MP4 files found in {output_dir}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files")
    print(f"Settings: fps={fps}, scale={scale}")
    print(f"{'='*60}")
    
    for mp4_path in mp4_files:
        gif_path = mp4_path.with_suffix('.gif')
        mp4_to_gif(str(mp4_path), str(gif_path), fps=fps, scale=scale)
    
    print(f"\n{'='*60}")
    print(f"✓ All GIFs created!")

if __name__ == '__main__':
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    process_output_folder(output_dir, fps=fps, scale=scale)
