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


