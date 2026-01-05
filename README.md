# 🐱🐶 Cat vs Dog Image Classification using CNN & Transfer Learning

## 📌 Overview
This project implements a **binary image classification system** to accurately distinguish between **cats and dogs** using **Convolutional Neural Networks (CNN)** and **Transfer Learning**.  
To enhance prediction accuracy and generalization, advanced deep learning techniques such as **data augmentation**, **pretrained models**, and **early stopping** were applied.

The trained model demonstrates strong performance on unseen images and is suitable for real-world image classification tasks.

---

## 🎯 Objectives
- Build a robust CNN-based image classification model  
- Leverage **transfer learning** for improved accuracy  
- Reduce overfitting using **data augmentation**  
- Optimize training with **early stopping**  
- Create an end-to-end **training and prediction pipeline**

---

## 🧠 Techniques & Concepts Used

### 1️⃣ Convolutional Neural Networks (CNN)
CNN layers automatically extract meaningful spatial features such as edges, textures, and shapes from input images, making them ideal for image classification tasks.

### 2️⃣ Transfer Learning
A **pretrained CNN model** is used as a feature extractor, allowing the model to benefit from previously learned image representations.  
This approach significantly reduces training time and improves overall performance.

### 3️⃣ Data Augmentation
Implemented using `ImageDataGenerator` to increase dataset diversity and improve generalization:
- Image rotation
- Zooming
- Width & height shifting
- Horizontal flipping  

This helps prevent overfitting and improves model robustness.

### 4️⃣ Early Stopping
Early stopping is applied during training to:
- Monitor validation loss
- Stop training when performance stops improving
- Retain the best-performing model weights  

This ensures optimal model performance without over-training.

### 5️⃣ Binary Classification
- Class mode: `binary`  
- Output labels:  
  - `0 → Cat`  
  - `1 → Dog`

---

## 🗂️ Project Structure
CAT_DOG_CLASSIFICATION/
├── dataset/
│ ├── train/
│ │ ├── cats/
│ │ └── dogs/
│ └── val/
│ ├── cats/
│ └── dogs/
├── model/
│ └── cat_dog_best.keras
├── plots/
│ └── training_plot.png
├── predictions/
│ └── test_images
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md

---

## ⚙️ Model Configuration
- Image Size: `224 × 224`
- Batch Size: `32`
- Loss Function: `Binary Crossentropy`
- Optimizer: `Adam`
- Evaluation Metric: `Accuracy`

Training and validation performance is visualized and saved for analysis.

---

## 🔍 Prediction Workflow
The prediction pipeline:
- Loads the trained `.keras` model
- Preprocesses input images
- Generates class predictions with confidence scores  

The model successfully classifies unseen images as **cat or dog**.

---

## 📊 Results
- High validation accuracy
- Reduced overfitting
- Stable and efficient training
- Reliable predictions on new images

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt   

### 2️⃣ Train the Model    
python train_model.py    

### 3️⃣ Run Predictions   
python predict.py   

### 4️⃣ Output

Predicts whether the image is Cat or Dog  
Displays confidence scores  
Saves prediction results in the predictions/ directory   

---


---

## 👩‍💻 Author   

**Inderpreet Kaur**    
Aspiring Data Scientist | Machine Learning Enthusiast     

📧 Email: inderpreetkaur0649@gmail.com    
🔗 LinkedIn: https://www.linkedin.com/in/inderpreet-kaur-613b1437b/     

Passionate about building practical machine learning projects using deep learning and transfer learning techniques.    

