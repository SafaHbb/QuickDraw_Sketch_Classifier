# Recognising Quick, Draw! Sketches with Deep Learning

**Introduction**

Sketch recognition is a difficult task in computer vision due to the abstract and messy nature of human-drawn images.  
This project explores how deep learning models can classify quick doodles from Google’s [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset.

We trained and compared five models — CNN, LeNet, AlexNet, ResNet18, and ConvNeXt — to recognise hand-drawn sketches of **apple**, **car**, **cat**, **dog**, and **flower**.  
Input images were originally 28×28 grayscale and resized to 128×128 RGB for compatibility with deeper architectures.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd38b150-ee45-4a2a-b5ec-73bcfca38bfa" width="500"/>
  <br>
  <em>Fig. 1. Quick, Draw! Sample Data</em>
</p>

### 📁 Dataset

- Subset of Google’s open-source [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset  
- Each sketch is a 28×28 grayscale image drawn in under 20 seconds  
- **Selected categories**: `apple`, `car`, `cat`, `dog`, `flower`  
- **600 samples per class** → 3,000 total images  
- Images normalised to [0–1], converted to RGB, and resized to **128×128**  
- Class labels encoded as integers (e.g. apple = 0, car = 1, etc.)

#### 🔄 Data Splits

- 80% for training  
- 20% for validation  
- From validation: 20% reused as a test set

#### 🧪 Preprocessing & Augmentation

**Training set:**
- Colour jitter (brightness, contrast, saturation)  
- Random horizontal flip  
- Random resized crop to 128×128  
- Normalisation (ImageNet mean & std)

**Validation/Test sets:**
- Resize to 128×128  
- Normalisation only
