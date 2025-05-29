# Recognising Quick, Draw! Sketches with Deep Learning

**Introduction:**  
Sketch recognition is a difficult task in computer vision due to the abstract and messy nature of human-drawn images.  
This project explores how deep learning models can classify quick doodles from Google’s [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.  
We trained and compared five models — CNN, LeNet, AlexNet, ResNet18, and ConvNeXt — to recognise hand-drawn sketches of **apple**, **car**, **cat**, **dog**, and **flower**.  
Input images were originally 28×28 grayscale and resized to 128×128 RGB for compatibility with deeper architectures.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd38b150-ee45-4a2a-b5ec-73bcfca38bfa" width="500"/>
  <br>
  <em>Fig. 1. Quick, Draw! Sample Data</em>
</p>
---  
**Dataset:**  
This project uses a 5-class subset of Google’s open-source [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset, which contains over 50 million hand-drawn sketches collected from an online game where users had 20 seconds to draw an object. Each sketch is a 28×28 grayscale image representing a single category.

**For this study:**  
_ **Selected classes**: `apple`, `car`, `cat`, `dog`, `flower`  
_ **Samples per class**: 600  
_ **Total images**: 3,000  
_ **Images were**:  
&nbsp;&nbsp;&nbsp;&nbsp;Normalised to the [0–1] range    
&nbsp;&nbsp;&nbsp;&nbsp;Converted to RGB  
&nbsp;&nbsp;&nbsp;&nbsp;Resized to 128×128 to support deeper models like AlexNet, ResNet, and ConvNeXt
    
_ **Labels encoded** as integers (e.g., `apple = 0`, `car = 1`, etc.)  
_ **Data Splits**:  
&nbsp;&nbsp;&nbsp;&nbsp;80% for training  
&nbsp;&nbsp;&nbsp;&nbsp;20% for validation  
&nbsp;&nbsp;&nbsp;&nbsp;From this, 20% was reused as a test set   

_ **Preprocessing & Augmentation**:  
&nbsp;&nbsp;&nbsp;&nbsp;Colour jitter (brightness, contrast, saturation)  
&nbsp;&nbsp;&nbsp;&nbsp;Random horizontal flip  
&nbsp;&nbsp;&nbsp;&nbsp;Random resized crop to 128×128  
&nbsp;&nbsp;&nbsp;&nbsp;Normalisation using ImageNet mean and std
    
_ **Validation/Test set transformations:**  
&nbsp;&nbsp;&nbsp;&nbsp;Resize to 128×128  
&nbsp;&nbsp;&nbsp;&nbsp;Normalisation only  

---
**Methodology:**  
This project evaluates five deep learning models for classifying hand-drawn sketches. All input images were resized to **128×128 RGB** to meet the input requirements of deeper models. Each model outputs class probabilities for: `apple`, `car`, `cat`, `dog`, and `flower`.  
  
1. Convolutional Neural Network (CNN)  
A simple CNN with two convolutional layers, ReLU activations, pooling layers, and fully connected output. It serves as a fast and lightweight baseline model.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9e5fe9a1-7125-4ea8-858f-52afaea5960b" alt="CNN Architecture" width="450"/>
  <br><em>Fig. 1 — Architecture of the basic CNN model</em>
</p>  

2. LeNet  
LeNet is an early CNN architecture adapted here for 128×128 RGB images. It uses average pooling and three fully connected layers. Performs well on simple data, but lacks depth.
  
<p align="center">
  <img src="images/lenet_arch.png" alt="LeNet Architecture" width="450"/>
  <br><em>Fig. 2 — Architecture of the LeNet model</em>
</p>  
  
3. AlexNet  
A deeper CNN pretrained on ImageNet with five convolutional and three dense layers. Fine-tuned by replacing the final layer for 5-class classification.

<p align="center">
  <img src="images/alexnet_arch.png" alt="AlexNet Architecture" width="450"/>
  <br><em>Fig. 3 — Architecture of the AlexNet model</em>
</p>

---

#### 📦 4. ResNet18

ResNet18 introduces residual connections to mitigate vanishing gradients. Pretrained and fine-tuned for this task. It delivers strong performance with efficient training.

<p align="center">
  <img src="images/resnet_arch.png" alt="ResNet18 Architecture" width="450"/>
  <br><em>Fig. 4 — Architecture of the ResNet18 model</em>
</p>

---

#### 📦 5. ConvNeXt

A modern CNN that combines convolutional layers with transformer-inspired design, including GELU activation and layer normalization. Achieved the best accuracy in this project.

<p align="center">
  <img src="images/convnext_arch.png" alt="ConvNeXt Architecture" width="450"/>
  <br><em>Fig. 5 — Architecture of the ConvNeXt model</em>
</p>


