# ConvNext Model for Classifying Alzeimers from Brain MRI scans
**Xander Akison (s4742991)**
### Table of Contents
- [ConvNext Model for Classifying Alzeimers from Brain MRI scans](#convnext-model-for-classifying-alzeimers-from-brain-mri-scans)
    - [Table of Contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Problem Outline](#problem-outline)
    - [Model](#model)
    - [Data Loading](#data-loading)
    - [Training](#training)
    - [Testing/Prediction](#testingprediction)
    - [Results](#results)
    - [Conclusion](#conclusion)
    - [References](#references)
    - [Dependencies](#dependencies)
### Introduction
Diagnosis through image classification of medical scans is becoming an increasingly more valuable use of pattern recognition tools in the modern world. With more and more accurate models arising constantly, the technology is seeing a greater adoption in clinical environments to help streamline health diagnosis and reduce costs. One of these developments is the adaption of traditional convolutional networks with newer Vision Transformer (ViT) models that allow for better global understanding from an input image. These new ConvNext models provide better regularisation across training data, improving test data accuracy.
### Problem Outline
This package seeks to implement a ConvNext Model capable of identifying Alzheimer's disease from labelled brain MRI scans contained in the ADNI dataset. The goal accuracy as provided in the ask is 80% or greater.
### Model
[`modules.py`](modules.py)
![ConvNext Architecture](images/ConvNext_Structure.png)
*Figure 1: ConvNext Architecture*
The designed model followed the standard practice structure. Incorporating four ConvNext Layers with the standard [3, 3, 9, 3] layout commonly seen in ConvNext-Tiny and ConvNext-Small applications. Where the balance of efficiency and power is paramount for classification success without spending large amounts of resources training too many weights. The larger third layer provides good mid-level feature extraction, something that is particularly useful in MRI image reasoning as it captures a lot of semantic abstraction, improving generalization.  
**Stem Layer**: The stem layer helps reduce spatial complexity by reducing the image height and width by a factor of four using a 2D convolution with a kernel and stride of four. Simultaneously, this layer takes the input dimension of one (greyscale) and creates enough output channels for the first ConvNext block (in this case 96 channels). The output is then normalized using layer norm before being passed into the series of computational blocks.
**ConvNext Block:** The model includes four ConvNext layers, each with double the channels as the previous block and a quarter of the image dimension space as the previous (from downsampling). The blocks follow the structure shown in Figure 2.
![ConvNext Block](images/ConvNext_Block.png)
*Figure 2: ConvNext Block*
The major differences to note between ConvNext and traditional ConvNet are:
- Change from standard Conv2d to Depthwise Conv2d: Groups the convolution by channels to reduce computation and capture broader context.
- Change from BatchNorm to LayerNorm: Improves stability for small batch sizes
- Activation uses GELU instead of ReLU
- Residual Scaling: The Gamma Scale step slowly increases the transformations effect on the residual outcome
- DropPath: Random chance of not using transform at all, helps improve regularization  

These changes allow for ConvNext models to generalize better on input by making each block behave more like a ViT without using attention.
### Data Loading
The ADNI data set can be downloaded from the [ADNI](https://adni.loni.usc.edu/) website. Although for the purposes of this report, the data was used directly from the University of Queensland's (UQ) rangpur cluster, where the train and test datasets were pulled from /home/groups/comp3710/ADNI. The 
### Training
### Testing/Prediction
### Results
### Conclusion
### References
https://www.researchgate.net/figure/The-architecture-of-the-ConvNeXt_fig4_361955951
### Dependencies