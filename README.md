# Recognising Quick, Draw! Sketches with Deep Learning

**⭐ Introduction:**  
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
**⭐ Dataset:**  
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
**⭐ Methodology:**  
This project evaluates five deep learning models for classifying hand-drawn sketches. All input images were resized to 128×128 RGB to meet the input requirements of deeper models. Each model outputs class probabilities for: `apple`, `car`, `cat`, `dog`, and `flower`.  
  
1. Convolutional Neural Network (CNN)  
A simple CNN with two convolutional layers, ReLU activations, pooling layers, and fully connected output. It serves as a fast and lightweight baseline model.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9e5fe9a1-7125-4ea8-858f-52afaea5960b" alt="CNN Architecture" width="450"/>
  <br><em>Fig. 1 — Architecture of the basic CNN model</em>
</p>  

2. LeNet  
LeNet is an early CNN architecture adapted here for 128×128 RGB images. It uses average pooling and three fully connected layers. Performs well on simple data, but lacks depth.
<p align="center">
  <img src="https://github.com/user-attachments/assets/b5de4a99-bffe-443f-8e11-1414e04012cb" alt="LeNet Architecture" width="450"/>
  <br><em>Fig. 2 — Architecture of the LeNet model</em>
</p>  
  
3. AlexNet  
A deeper CNN pretrained on ImageNet with five convolutional and three dense layers. Fine-tuned by replacing the final layer for 5-class classification.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/6e0afaa2-fd46-408c-9c01-87aca926700e" width="450"/>
  <br><em>Fig. 3 — Architecture of the AlexNet model</em>
</p>
  
4. ResNet18  
ResNet18 introduces residual connections to mitigate vanishing gradients. Pretrained and fine-tuned for this task. It delivers strong performance with efficient training.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/e247c540-3dc2-435e-b0a4-63fddd90b719" width="450"/>
  <br><em>Fig. 4 — Architecture of the ResNet18 model</em>
</p>  
  
5. ConvNeXt  
A modern CNN that combines convolutional layers with transformer-inspired design, including GELU activation and layer normalization. Achieved the best accuracy in this project.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/eb6a7140-e389-415e-848d-0107206e2fd3" alt="ConvNeXt Architecture" width="450"/>
  <br><em>Fig. 5 — Architecture of the ConvNeXt model</em>
</p>

---
**⭐ Loss Function:**  
All models were trained using **Cross Entropy Loss**, which compares predicted class probabilities with the true class label:   
Loss = − ∑ ( *yₙ* × log(*ŷₙ*) )   
**Loss** = − ∑<sub>i=1</sub><sup>C</sup> ( *y<sub>i</sub>* × log(*ŷ<sub>i</sub>* ) )  
Loss = - ∑_{i=1}^{C} (y_i * log(ŷ_i))  
Loss = - ∑_{i=1}^{C} (y_i * log(ŷ_i))  

Loss = $$\left( \sum_{k=1}^n b_k^2 \right)$$   

Loss = $$\left( \sum_{i=1}^C y_i * log(*ŷₙ* \right)^2  

$$
\text{Loss} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$  

$$\left text{Loss} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)\right$$
 

Where:  
- *C* = number of classes (5)  
- *yₙ* = 1 if class *i* is correct, 0 otherwise  
- *ŷₙ* = predicted probability for class *i* (after softmax)
  
---
**⭐ Optimisation:**  
All models used the same training setup with Stochastic Gradient Descent (SGD):  
&nbsp;&nbsp;&nbsp;&nbsp;Optimizer: SGD  
&nbsp;&nbsp;&nbsp;&nbsp;Learning Rate: 0.01  
&nbsp;&nbsp;&nbsp;&nbsp;Batch Size: 32  
&nbsp;&nbsp;&nbsp;&nbsp;Epochs: 10  

A GPU was used to speed up training for deeper models like AlexNet, ResNet18, and ConvNeXt. The best model (based on validation accuracy) was saved and evaluated on the test set.

---
**⭐ Evaluation Metrics:**   
Four common metrics were used to evaluate model performance:  
&nbsp;&nbsp;&nbsp;&nbsp;Accuracy = (TP + TN) / (TP + TN + FP + FN)  
&nbsp;&nbsp;&nbsp;&nbsp;Precision = TP / (TP + FP)  
&nbsp;&nbsp;&nbsp;&nbsp;Recall = TP / (TP + FN)  
&nbsp;&nbsp;&nbsp;&nbsp;F1 Score = 2 * (Precision * Recall) / (Precision + Recall)  

Where:  
TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives  

---

