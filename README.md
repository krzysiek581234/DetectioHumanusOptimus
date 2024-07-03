# Convolutional Neural Networks in Classification of Faces

## Table of Contents

1. [Overview](#overview)
2. [Project Objectives](#project-objectives)
3. [Data](#data)
4. [Methodology](#methodology)
5. [Architecture](#architecture)
6. [Implementation](#implementation)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [References](#references)

<a name="overview"></a>
## 1. Overview
This project focuses on the development and evaluation of a convolutional neural network (CNN) for classifying human faces based on images. The main objective is to accurately identify human faces in various images using a CNN.

<a name="project-objectives"></a>
## 2. Project Objectives
- Implement a convolutional neural network for face classification.
- Evaluate the performance of the CNN against other neural network architectures and transfer learning solutions.
- Analyze the effectiveness of different optimization algorithms.
- Address the issue of an imbalanced dataset using weighted cross-entropy loss.

<a name="data"></a>
## 3. Data
The training dataset includes:
- 64,770 facial images.
- 26,950 non-representational images.
- All images are 36x36 pixels in grayscale.

<a name="methodology"></a>
## 4. Methodology
1. **Supervised Learning**: Using the XYZ training dataset.
2. **Optimizers**: Comparison between Stochastic Gradient Descent (SGD) and Adam.
3. **Handling Imbalanced Dataset**: Using weighted cross-entropy loss.
4. **Data Augmentation**: Applying various transformations to increase data diversity.

<a name="architecture"></a>
## 5. Architecture
### Default Network
- **Convolutional Layers**: 
  - First: 1 input, 6 outputs, 5x5 kernel.
  - Second: 6 inputs, 16 outputs, 5x5 kernel.
- **Pooling Layers**: Max pooling with a 2x2 window.
- **Fully Connected Layers**: 
  - First: 16\*6\*6 inputs, 32 outputs.
  - Second: 32 inputs, 16 outputs.
  - Third: 16 inputs, 2 outputs.

### Custom Network
- **Convolutional Layers**: 
  - First: 1 input, 8 outputs, 5x5 kernel.
  - Second: 8 inputs, 16 outputs, 3x3 kernel.
  - Third: 16 inputs, 32 outputs, 3x3 kernel.
  - Fourth: 32 inputs, 64 outputs, 3x3 kernel.
- **Batch Normalization**: After each convolutional layer.
- **Activation Functions**: ReLU and LeakyReLU.
- **Pooling Layers**: Max pooling after the second, third, and fourth convolutional layers.
- **Fully Connected Layers**: 
  - First: 576 inputs, 256 outputs.
  - Second: 256 inputs, 128 outputs.
  - Third: 128 inputs, 2 outputs.

<a name="implementation"></a>
## 6. Implementation
- **Training and Validation**: Early stopping and best model saving based on validation performance.
- **Data Augmentation**: RandomPosterize, RandomHorizontalFlip, RandomVerticalFlip, RandomEqualize, RandomInvert, RandomAffine, GaussianBlur, ColorJitter.

<a name="results"></a>
## 7. Results
- **Optimizer Performance**: Adam performed better for face image classification.
- **Imbalanced Dataset Handling**: Weight scaling improved model performance by 2 percentage points.
- **Data Augmentation**: Enhanced the diversity of training data and improved model performance.

<a name="conclusion"></a>
## 8. Conclusion
The convolutional neural network developed for this project demonstrated effective face classification capabilities. Adam optimizer showed superior performance, and handling the imbalanced dataset through weighted cross-entropy significantly improved results. Data augmentation played a crucial role in enhancing model performance.

<a name="references"></a>
## 9. References
- Project repository: [Face Classification Project](https://github.com/krzysiek581234/France_AI_YOLO)
- YOLO v8: [YOLO v8 Repository](https://github.com/ultralytics/ultralytics)
