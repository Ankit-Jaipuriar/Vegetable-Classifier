# Vegetable Image Classification

This project performs **vegetable classification** using **Transfer Learning** with **MobileNetV2** on a custom dataset of vegetable images.

## 📂 Dataset Preparation
- The dataset is stored on **Google Drive** under the path:  
  `/content/drive/MyDrive/VegetableImages`
- It is copied to the Colab environment for faster access:
  ```
  /content/vegetable_dataset/
    ├── train/
    ├── validation/
    └── test/
  ```
- Each folder contains images sorted into class subdirectories.

## 🔧 Setup and Data Augmentation
- `ImageDataGenerator` is used to:
  - Normalize pixel values.
  - Apply random transformations **only** to the training data (rotation, shift, shear, zoom, flip).

## 📊 Data Visualization
- **Class Distribution** of training data is shown using bar plots.
- **Sample Images** from the first three classes are displayed.

## 🧠 Model Building
- **Transfer Learning** using **MobileNetV2** (pre-trained on ImageNet) as the base model.
- The base model is **frozen** (not trainable).
- The custom head consists of:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - Final `Dense` layer with **softmax** activation.

- **Compilation**:
  - Optimizer: `Adam` (learning rate = 1e-4)
  - Loss: `categorical_crossentropy`
  - Metric: `accuracy`

## 🏋️ Training
- **Callbacks** used:
  - `EarlyStopping` (patience=3, restore best weights)
  - `ModelCheckpoint` (save the best model based on validation accuracy)
- **Training Duration**: 10 epochs

## 📈 Evaluation
- Final model is evaluated on the **test set**.
- **Accuracy and loss** progress is plotted for both training and validation data.

## 🖼️ Single Image Prediction
- Load and preprocess a single image.
- Predict its class and visualize the result.

## 🔎 Multiple Sample Predictions
- Predict and visualize a few random images from the **test dataset**.

## 📋 Additional Evaluation
- **Confusion Matrix** for all classes.
- **Classification Report** with precision, recall, and F1-score for each class.

---

## 🚀 How to Run
1. Open the Colab notebook.
2. Mount your Google Drive.
3. Copy the dataset to local Colab storage.
4. Run all cells sequentially.

---

## 📁 Outputs
- Final trained model saved as:  
  `vegetable_classifier_model_final2.h5`

---
