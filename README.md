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

$$\text{Loss} = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)$$  

Where:  
C is the number of classes (in our case, 5).  
y<sub>i</sub> is 1 if class i is the correct class, and 0 otherwise.   
ŷ<sub>i</sub> is the predicted probability for class i (after softmax).  

---
**⭐ Optimisation:**  
All models used the same training setup with Stochastic Gradient Descent (SGD):  
  
Optimizer: SGD  
Learning Rate: 0.01  
Batch Size: 32  
Epochs: 10  

A GPU was used to speed up training for deeper models like AlexNet, ResNet18, and ConvNeXt. The best model (based on validation accuracy) was saved and evaluated on the test set.

---
**⭐ Evaluation Metrics:**   
Four common metrics were used to evaluate model performance:  
  
Accuracy = (TP + TN) / (TP + TN + FP + FN)  
Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)  
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)  

Where:  
TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives  

---

**⭐ Results:**  
This section presents both quantitative and qualitative evaluation results to assess how well each model performed in recognising hand-drawn sketches from the Quick, Draw! dataset.

#### 🌱 Quantitative Evaluation

Each model was trained for 10 epochs and evaluated using the following metrics:

- **Accuracy** – How many predictions were correct overall  
- **Precision** – Of the predicted labels, how many were correct  
- **Recall** – Of the actual labels, how many were detected correctly  
- **F1 Score** – Harmonic mean of precision and recall, balancing both

🔸ConvNeXt outperformed all other models across every metric, achieving 95% accuracy and showing excellent balance between precision and recall.  
🔸AlexNet and ResNet18 also performed strongly (~91.6% accuracy), confirming the benefit of deeper architectures.  
🔸CNN and LeNet showed lower accuracy, struggling to generalise complex or abstract sketch features.  

**Evaluation Summary:**
| Model     | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| CNN       | 0.775    | 0.7688    | 0.7892 | 0.7683   |
| LeNet     | 0.7083   | 0.7469    | 0.7282 | 0.7026   |
| AlexNet   | 0.9167   | 0.9246    | 0.9187 | 0.9188   |
| ResNet18  | 0.9167   | 0.9218    | 0.9187 | 0.9184   |
| ConvNeXt  | **0.9500** | **0.9548** | **0.9526** | **0.9507** |  

  

🔸This chart shows that ConvNeXt achieved the best overall performance across all four metrics: accuracy, precision, recall, and F1 score.
🔸AlexNet and ResNet18 also performed very well, with similar high scores across all metrics.
🔸On the other hand, CNN and LeNet showed weaker results, particularly in recall and F1 score, indicating more frequent misclassifications or imbalance in prediction confidence.      
<p align="center">
  <img src="https://github.com/user-attachments/assets/ea2e5131-fbc2-4500-b911-e75ce1d1775f" alt="Training Loss Curves" width="500"/>
  <br><em>Fig. 1 — Loss curves showing learning progress across models</em>
</p>  

  
The training loss curves show how each model learned over 10 epochs.  
🔸ConvNeXt and ResNet18 had the fastest and smoothest convergence, meaning they learned the patterns in the data efficiently and consistently.  
🔸AlexNet also converged quickly, with slightly higher loss than ConvNeXt.  
🔸CNN and LeNet were slower to converge and less stable, which reflects their lower capacity to model abstract sketch features and generalise well.   
<p align="center">
  <img src="https://github.com/user-attachments/assets/48b66b02-8f35-4089-9e30-ee78ecc43789" alt="ConvNeXt Prediction Examples" width="500"/>
  <br><em>Fig. 2 — Top: Correct predictions; Bottom: Incorrect predictions</em>
</p>  



#### 🌱 Qualitative Results:  

To better understand the model's behaviour, sample predictions from ConvNeXt (the best-performing model) were analysed.  
🔸The top row shows correct predictions, even for sketches with unusual or abstract shapes, indicating strong generalisation.  
🔸The bottom row shows incorrect predictions, such as misclassifying a “cat” as a “dog” or a “flower” as a “dog.” These mistakes often occurred due to visual similarity or overly simplistic drawings.  
🔸While ConvNeXt performed well overall, it occasionally struggled with noisy or ambiguous sketches — especially when different categories had similar outlines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a7ece7b-db8f-46c6-9bc6-3b5630e76eee" alt="ConvNeXt Prediction Examples" width="500"/>
  <br><em>Fig. 2 — Examples of ConvNeXt Predictions. (Top: Correct predictions (green), Bottom: Incorrect predictions (red) with true class in parentheses.)</em>
</p>  


---
#### 🏆 Final Comparison & Insights  

🔸**ConvNeXt** demonstrated the highest performance and robustness across both metrics and real-world sketch variability.  
🔸**AlexNet** and **ResNet18** also performed well, confirming that deeper networks with pretrained weights adapt effectively to noisy, abstract input.  
🔸**CNN** and **LeNet** were limited by their simpler architecture and struggled particularly with edge cases and similar-shaped objects.

These findings confirm that **modern, deeper architectures generalise better** and are more suitable for sketch recognition tasks involving human-drawn, inconsistent data.


