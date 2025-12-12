


# STCN + YOLOv8 Video Object Detection

<p align="center">
	<img src="output/baseball.gif" width="220" alt="Baseball Example" />
	<img src="output/beach.gif" width="220" alt="Beach Example" />
	<img src="output/city_ride.gif" width="220" alt="City Ride Example" />
	<img src="output/horse.gif" width="220" alt="Horse Example" />
</p>

<p align="center">
	<b>Example Results: Baseball | Beach | City Ride | Horse</b>
</p>

Fast and accurate object detection for video using STCN and YOLOv8 on GPU. Each object is highlighted with a bounding box and class label.

## Features

- ✅ Real-time video frame processing
- ✅ YOLOv8 object detection
- ✅ GPU acceleration (NVIDIA CUDA)
- ✅ Customizable object detection (choose classes)
- ✅ Bounding boxes and class labels for each detected object


## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support
- PyTorch with CUDA
- OpenCV
- YOLOv8 (ultralytics)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command
```bash
python stcn_object_segmentation.py <frame_dir> [output_dir] [object1,object2,...]
```

### Examples

**Detect ALL COCO objects:**
```bash
python stcn_object_segmentation.py dataset/horse/horses-kids output/result
```

**Detect specific objects:**
```bash
python stcn_object_segmentation.py dataset/horse/horses-kids output/result person,horse
python stcn_object_segmentation.py dataset/traffic output/result car,bus,truck
python stcn_object_segmentation.py dataset/beach output/result person,surfboard,dog
```

## Parameters

- `<frame_dir>` - Directory containing frame images (.jpg, .png)
- `[output_dir]` - Output directory for segmentation videos (default: `output/stcn_detection`)
- `[object1,object2,...]` - Optional comma-separated list of objects to detect. Leave empty to detect all COCO classes

## Supported Objects

All 80 COCO classes including: person, car, dog, cat, horse, bicycle, truck, bus, motorcycle, airplane, boat, train, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush, and more


## Output Files

Generated in `<output_dir>/`:
- `stcn_labeled_overlay.mp4` - Original frames with bounding boxes and class labels
- `stcn_labeled_mask.mp4` - (Optional) Masked output if enabled in code

## Performance

- **GPU**: NVIDIA RTX 4060+ (8GB+ VRAM)
- **Speed**: ~9-10 fps for 4K video
- **Accuracy**: High-quality instance segmentation with YOLOv8


## How It Works

1. **Input**: Frame sequence from video or image directory
2. **Detection**: YOLOv8 detects target objects in each frame
3. **Visualization**: Draws bounding boxes and class labels on each detected object
4. **Output**: Generates MP4 video with detection results

## GPU Requirements

- NVIDIA GPU (CUDA compute capability 3.0+)
- CUDA Toolkit 11.8+
- cuDNN 8.0+

## Troubleshooting

**Out of memory error:**
- Reduce frame resolution in input
- Use fewer objects to detect

**Low detection rate:**
- Lower confidence threshold in code
- Try different object names

**Slow processing:**
- Reduce video resolution
- Use GPU with more VRAM
- `comparison.mp4` - Side-by-side comparison (original | overlay | segmentation)

## Model

Uses pre-trained DeepLabV3 ResNet50 from PyTorch model zoo:
- Pre-trained on COCO (80 classes)
- Fine-tuned on Pascal VOC (21 classes)
- Automatic download on first run

## Detected Classes

The model can detect 21 object classes:
- People, animals (horse, dog, cat, bird, etc.)
- Vehicles (car, bus, train, bicycle, motorbike, etc.)
- Indoor objects (chair, sofa, table, etc.)


## Example Results

**Horse Video (78 frames)**:
- Detected classes: person, horse
- Processing time: 1.1s/frame
- Average confidence: 93.5%
