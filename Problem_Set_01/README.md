# Problem Set 01: Chest X-Ray Pneumonia Classification

## 1. Objective
Develop a Convolutional Neural Network (CNN) model capable of accurately classifying pediatric chest X-ray images into **Normal** or **Pneumonia**.

## 2. Approach & Data Pipeline
- **Preprocessing**: Rescaling pixel intensity values from `[0, 255]` to `[0, 1]`. Images are resized to `150x150` pixels.
- **Data Augmentation**: To prevent overfitting and enhance generalization, training images undergo random rotations (up to 15°), horizontal flips, and spatial zooms/shifts.
- **Batching**: Processed in mini-batches of size 32.

## 3. Network Architecture
- **Feature Extractors**: 4 Convolutional blocks (3x3 kernels) paired with **Batch Normalization** for training stability and **MaxPooling** (2x2) for spatial downsampling.
- **Regularization**: Dropout (0.5) applied prior to the output layer to reduce over-reliance on individual feature representations.
- **Output Layer**: Single Dense unit with a **Sigmoid** activation function for binary decision output (`0 = Normal`, `1 = Pneumonia`).

## 4. Optimization & Methodology
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam (Learning rate = `1e-4`).
- **Callbacks**: Early Stopping with weight restoration to halt training if validation loss plateaus for 5 consecutive epochs.

## 5. Findings & Evaluation Metric Selection
In pediatric chest X-ray diagnosis, **Recall** is prioritized over overall Accuracy to minimize **False Negatives** (misdiagnosing a pneumonia patient as normal). 

- **Test Accuracy**: ~88% - 92%
- **Test Recall**: High sensitivity (>92%) ensures most actual pneumonia cases are caught.
