# Indoor Climbing Hold Detection

A computer vision system for detecting and analyzing climbing holds on indoor bouldering walls using color segmentation, texture analysis, and shape features.

## Overview

This project implements a multi-feature approach to automatically detect climbing holds in images of indoor climbing walls. Unlike simple color-based detection systems, this implementation combines:

- **Color Segmentation**: HSV-based color detection with special handling for challenging colors like red
- **Texture Analysis**: Local Binary Patterns (LBP) and Gray-Level Co-occurrence Matrix (GLCM) features
- **Shape Analysis**: Geometric properties like compactness, solidity, and aspect ratio
- **Classification System**: Rule-based scoring to distinguish actual holds from false positives

## Features

### Core Functionality
- Multi-color hold detection with customizable color targets
- Robust texture feature extraction (with fallback methods)
- Comprehensive shape analysis using contour properties
- Confidence-based classification system
- Interactive color picker for precise color selection
- Detailed analysis and visualization of results

### Advanced Features
- Automatic dominant color extraction from images
- HSV color space processing with red wraparound handling
- Morphological operations for noise reduction
- Alternative texture measures when GLCM is unavailable
- Configurable classification rules for different gym setups

**Note**: The system works with fallback texture measures if scikit-image is not available.

## Usage

### Basic Analysis
```python
# Analyze climbing wall with automatic color detection
classified_holds, results = enhanced_climbing_analysis('your_climbing_wall.jpg')
```

### Custom Color Targeting
```python
# Specify exact hold colors (RGB format)
custom_colors = [
    [220, 20, 20],   # Red holds
    [20, 200, 20],   # Green holds
    [20, 20, 220],   # Blue holds
]

classified_holds, results = enhanced_climbing_analysis(
    'climbing_wall.jpg', 
    target_colors=custom_colors
)
```

### Advanced Configuration
```python
# Custom classification rules for your specific gym
custom_rules = {
    'min_area': 300,           # Minimum hold size (pixels)
    'max_area': 30000,         # Maximum hold size (pixels)
    'max_compactness': 3.0,    # Shape constraint (lower = more circular)
    'min_solidity': 0.4,       # Minimum filled ratio
    'max_aspect_ratio': 3.0,   # Width/height ratio limit
    'min_extent': 0.3,         # Minimum bbox fill ratio
    'max_color_distance': 80,  # Color matching tolerance
    'min_edge_density': 0.02,  # Minimum texture requirement
}

classified_holds, results = enhanced_climbing_analysis(
    'image.jpg',
    target_colors=custom_colors,
    classification_rules=custom_rules
)
```

### Interactive Color Selection
```python
# Load image and pick colors interactively
img, img_rgb = load_and_display_image('climbing_wall.jpg')
interactive_color_picker(img_rgb)

# Get exact RGB values at specific coordinates
color = get_pixel_color(img_rgb, x=100, y=200)
```

### Debugging Color Detection
```python
# Test different tolerance values for problematic colors
debug_color_detection(img_rgb, [255, 0, 0], tolerance_range=[30, 50, 70, 90])
```

## Results Interpretation

The system classifies each detected region into three categories:

### Classification Levels
- **🟢 Likely Hold** (6-8 points): High confidence detections with good color, shape, and texture
- **🟡 Possible Hold** (4-5 points): Moderate confidence, may need manual verification
- **🔴 Unlikely Hold** (0-3 points): Probable false positives (shadows, wall features, etc.)

### Feature Analysis
Each detection includes detailed feature analysis:
```python
for hold in results['Likely Hold']:
    print(f"Hold at {hold['center']}")
    print(f"Confidence: {hold['confidence_score']}/8")
    print(f"Area: {hold['area']:.0f} pixels")
    print(f"Shape quality: {hold['compactness']:.2f}")
    print(f"Color match: {hold['color_distance']:.1f}")
    print(f"Texture: {hold['edge_density']:.3f}")
```

## Project Structure

```
climbing-hold-detection/
├── climbing_hold_segmentation.py    # Main analysis code
├── README.md                        # This file
├── examples/                        # Example images and usage
├── results/                         # Output visualizations
└── docs/                           # Additional documentation
```

## Technical Details

### Color Detection
- Uses HSV color space for better lighting robustness
- Special handling for red color wraparound (0°/360°)
- Morphological operations for noise cleanup
- Configurable tolerance parameters

### Texture Features
- Local Binary Patterns (LBP) for surface texture
- Gray-Level Co-occurrence Matrix (GLCM) for texture properties

**Fallback methods:**
- Gradient-based contrast measures
- Edge density analysis
- Local variance computation
- Intensity range analysis

### Shape Features
- Contour area and perimeter
- Compactness (circularity measure)
- Aspect ratio and bounding box extent
- Solidity (convex hull ratio)
- Hu moments for rotation invariance

## Limitations and Considerations

### Current Limitations
- Requires manual color selection for best results
- Performance depends on lighting conditions
- May struggle with very small or heavily shadowed holds
- Single-image analysis (no temporal information)

### Areas for Improvement
- Machine learning-based hold detection
- Route reconstruction and difficulty estimation
- Implementing/Fixing Gray Level Co-occurrence Matrix

## Acknowledgments

- [Indoor Climbing Hold and Route Segmentation](https://github.com/xiaoxiae/Indoor-Climbing-Hold-and-Route-Segmentation) - University of Heidelberg
- [ClimbNet](https://github.com/cydivision/climbnet) - Detectron2-based approach
- [Route Classification](https://github.com/tonylay7/bouldering_route_classification) - ML-based route grading
