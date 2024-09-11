# Object Detection with YOLO (You Only Look Once)

This project demonstrates how to implement the YOLO (You Only Look Once) object detection model, specifically the TinyYOLOv2 architecture, using the PASCAL VOC dataset. YOLO is a fast and accurate object detection algorithm that predicts bounding boxes and class probabilities for objects in a single forward pass through the network. In this project, we will:

- Load and preprocess the PASCAL VOC dataset
- Define and build the TinyYOLOv2 model in PyTorch
- Visualize object detection predictions with bounding boxes
- Load pre-trained model weights
- Apply non-maximum suppression (NMS) to filter the best predictions

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Model Training and Inference](#model-training-and-inference)
5. [Results](#results)
6. [Acknowledgments](#acknowledgments)

---

## Installation

To run this project, you need the following dependencies:

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Pillow

Install the required libraries by running:

```bash
pip install torch torchvision matplotlib Pillow
```

For additional model summary visualization:

```bash
pip install torch-summary
```

---

## Dataset

We are using the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), which contains images and annotations for 20 object classes, such as aeroplane, bicycle, bird, etc. The dataset is downloaded and extracted automatically in this project.

The classes in the VOC dataset are:

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
```

### Download the Dataset
If the dataset is not already available locally, it will be downloaded and extracted automatically:

```python
import os
import urllib.request
import tarfile

if not os.path.exists("VOCdevkit"):
    if not os.path.exists("VOC.tar"):
        urllib.request.urlretrieve(
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", "VOC.tar"
        )
    with tarfile.open("VOC.tar") as tar:
        tar.extractall()
```

---

## Model Architecture

This project implements the TinyYOLOv2 model, which is a lightweight version of the YOLO model. The architecture consists of convolutional and max-pooling layers, with batch normalization and Leaky ReLU activations. The model predicts bounding boxes and class probabilities simultaneously.

Key layers include:

1. **Convolutional Layers**: Nine convolutional layers with increasing filter sizes.
2. **Max Pooling Layers**: Six max-pooling layers with stride 2 to downsample the feature map.
3. **Final Layer**: A 1x1 convolutional layer to output the bounding boxes and class predictions.

Here is a summary of the architecture:

| Layer                | Filters | Kernel Size | Stride | Padding | Activation  |
|----------------------|---------|-------------|--------|---------|-------------|
| Conv1 + BN + LeakyReLU| 16      | 3x3         | 1      | 1       | Leaky ReLU  |
| MaxPool1              | -       | 2x2         | 2      | -       | -           |
| Conv2 + BN + LeakyReLU| 32      | 3x3         | 1      | 1       | Leaky ReLU  |
| MaxPool2              | -       | 2x2         | 2      | -       | -           |
| ...                   | ...     | ...         | ...    | ...     | ...         |
| Conv9 (Output Layer)  | 425     | 1x1         | 1      | 0       | Linear      |

---

## Model Training and Inference

The model is pre-trained on the VOC dataset, and the pre-trained weights can be downloaded using the following:

```python
if not os.path.exists("yolov2-tiny-voc.weights"):
    urllib.request.urlretrieve(
        "https://pjreddie.com/media/files/yolov2-tiny-voc.weights",
        "yolov2-tiny-voc.weights",
    )
```

### Visualizing Predictions

The trained TinyYOLOv2 model predicts bounding boxes and object classes in real-time. You can visualize the predictions as follows:

```python
input_tensor = load_image_batch([33], 320)  # Load an example image
output_tensor = network(input_tensor)       # Run the model to get predictions
show_images_with_boxes(input_tensor, output_tensor)  # Display predictions with bounding boxes
```

---

## Results

After running the YOLO model, you will see object detection results with bounding boxes around objects in the images. You can adjust the confidence threshold and apply non-maximum suppression (NMS) to filter out low-confidence boxes and overlapping boxes.

---

## Acknowledgments

This project is inspired by the original YOLO paper and [pjreddie's Darknet](https://github.com/pjreddie/darknet). The implementation is done in PyTorch with the use of the torchvision library for image preprocessing and visualization.

