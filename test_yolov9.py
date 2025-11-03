"""
Simple YOLOv9 Test Script for Climbing Hold Detection
Quick testing on a minimal dataset (5-10 images)
"""

import os
import torch
from pathlib import Path

def setup_minimal_test():
    """
    Setup for minimal YOLOv9 testing
    """
    print("=" * 70)
    print("MINIMAL YOLOV9 TEST SETUP")
    print("=" * 70)
    
    # Step 1: Clone YOLOv9
    print("\n[1/5] Cloning YOLOv9 repository...")
    if not Path('yolov9').exists():
        os.system('git clone https://github.com/WongKinYiu/yolov9.git')
        print("✓ YOLOv9 cloned")
    else:
        print("✓ YOLOv9 already exists")
    
    # Step 2: Install requirements
    print("\n[2/5] Installing requirements...")
    os.system('pip install -r yolov9/requirements.txt --break-system-packages')
    os.system('pip install opencv-python matplotlib pyyaml --break-system-packages')
    print("✓ Requirements installed")
    
    # Step 3: Create minimal dataset structure
    print("\n[3/5] Creating dataset structure...")
    dirs = [
        'test_dataset/images/train',
        'test_dataset/images/val',
        'test_dataset/labels/train',
        'test_dataset/labels/val'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Dataset structure created")
    
    # Step 4: Create data.yaml
    print("\n[4/5] Creating configuration file...")
    yaml_content = f"""
# Climbing Holds Dataset Configuration
path: {Path('test_dataset').absolute()}
train: images/train
val: images/val

# Classes
names:
  0: climbing_hold

# Number of classes
nc: 1
"""
    with open('test_dataset/data.yaml', 'w') as f:
        f.write(yaml_content)
    print("✓ Configuration file created")
    
    # Step 5: Download pretrained weights
    print("\n[5/5] Downloading pretrained weights...")
    if not Path('yolov9/yolov9-s.pt').exists():
        os.system('wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt -O yolov9/yolov9-s.pt')
        print("✓ Weights downloaded")
    else:
        print("✓ Weights already exist")
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print_next_steps()


def print_next_steps():
    """Print instructions for next steps"""
    print("\n📋 NEXT STEPS:")
    print("-" * 70)
    print("\n1️⃣  PREPARE YOUR DATA:")
    print("   • Add 5-10 climbing wall images to: test_dataset/images/train/")
    print("   • Add 2-3 validation images to: test_dataset/images/val/")
    print("   • Name images: img1.jpg, img2.jpg, etc.")
    
    print("\n2️⃣  CREATE LABELS (YOLO format):")
    print("   • For each image, create a .txt file with the same name")
    print("   • Format: <class_id> <x_center> <y_center> <width> <height>")
    print("   • All values normalized to [0-1]")
    print("   • Example label file 'img1.txt':")
    print("     0 0.5 0.3 0.1 0.15")
    print("     0 0.7 0.6 0.08 0.12")
    print("   • Save to: test_dataset/labels/train/ and test_dataset/labels/val/")
    
    print("\n3️⃣  QUICK ANNOTATION GUIDE:")
    print("   For a hold at pixel coordinates (x1, y1) to (x2, y2):")
    print("   x_center = (x1 + x2) / (2 * image_width)")
    print("   y_center = (y1 + y2) / (2 * image_height)")
    print("   width = (x2 - x1) / image_width")
    print("   height = (y2 - y1) / image_height")
    
    print("\n4️⃣  START TRAINING:")
    print("   Run: python test_yolov9.py --train")
    
    print("\n5️⃣  TEST INFERENCE:")
    print("   Run: python test_yolov9.py --detect <image_path>")
    print("-" * 70)


def train_minimal():
    """
    Train YOLOv9 on minimal dataset
    """
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    # Check if data exists
    train_images = list(Path('test_dataset/images/train').glob('*.jpg')) + \
                   list(Path('test_dataset/images/train').glob('*.png'))
    
    if len(train_images) == 0:
        print("❌ ERROR: No training images found!")
        print("Please add images to test_dataset/images/train/")
        return
    
    print(f"\n✓ Found {len(train_images)} training images")
    
    # Training parameters for small dataset
    epochs = 100  # More epochs for small dataset
    batch_size = 4  # Small batch for small dataset
    img_size = 640
    
    train_cmd = f"""
    cd yolov9 && python train.py \\
        --batch {batch_size} \\
        --epochs {epochs} \\
        --img {img_size} \\
        --data ../test_dataset/data.yaml \\
        --weights yolov9-s.pt \\
        --device {0 if torch.cuda.is_available() else 'cpu'} \\
        --name climbing_holds_test \\
        --cache
    """
    
    print(f"\nTraining with:")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Image size: {img_size}")
    print(f"  • Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("\nStarting training (this may take a while)...\n")
    
    os.system(train_cmd)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nResults saved to: yolov9/runs/train/climbing_holds_test/")
    print("Best weights: yolov9/runs/train/climbing_holds_test/weights/best.pt")
    print("\nTo test detection, run:")
    print("  python test_yolov9.py --detect <image_path>")


def detect(image_path, weights_path=None):
    """
    Run detection on an image
    """
    if weights_path is None:
        weights_path = 'yolov9/runs/train/climbing_holds_test/weights/best.pt'
    
    if not Path(weights_path).exists():
        print(f"❌ ERROR: Weights not found at {weights_path}")
        print("Please train the model first: python test_yolov9.py --train")
        return
    
    print(f"\n🔍 Running detection on: {image_path}")
    print(f"Using weights: {weights_path}")
    
    detect_cmd = f"""
    cd yolov9 && python detect.py \\
        --weights ../{weights_path} \\
        --source ../{image_path} \\
        --conf 0.25 \\
        --name test_detection
    """
    
    os.system(detect_cmd)
    
    print("\n✓ Detection complete!")
    print("Results saved to: yolov9/runs/detect/test_detection/")


def validate(weights_path=None):
    """
    Validate model on validation set
    """
    if weights_path is None:
        weights_path = 'yolov9/runs/train/climbing_holds_test/weights/best.pt'
    
    if not Path(weights_path).exists():
        print(f"❌ ERROR: Weights not found at {weights_path}")
        return
    
    print(f"\n📊 Running validation...")
    
    val_cmd = f"""
    cd yolov9 && python val.py \\
        --weights ../{weights_path} \\
        --data ../test_dataset/data.yaml \\
        --img 640 \\
        --name test_validation
    """
    
    os.system(val_cmd)
    
    print("\n✓ Validation complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Setup:       python test_yolov9.py --setup")
        print("  Train:       python test_yolov9.py --train")
        print("  Detect:      python test_yolov9.py --detect <image_path>")
        print("  Validate:    python test_yolov9.py --validate")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == '--setup':
        setup_minimal_test()
    
    elif command == '--train':
        train_minimal()
    
    elif command == '--detect':
        if len(sys.argv) < 3:
            print("❌ ERROR: Please provide image path")
            print("Usage: python test_yolov9.py --detect <image_path>")
            sys.exit(1)
        detect(sys.argv[2])
    
    elif command == '--validate':
        validate()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use --setup, --train, --detect, or --validate")