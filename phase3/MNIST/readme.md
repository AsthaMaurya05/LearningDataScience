# 🧠 MNIST Handwritten Digit Classification using ANN (TensorFlow & Keras)

## 🚀 Project Overview

This project focuses on classifying handwritten digits (0–9) from the MNIST dataset using an Artificial Neural Network (ANN).

Instead of directly training a neural network, I followed a complete Deep Learning workflow:

```python
Dataset Understanding → Visualization → Preprocessing →
ANN Design → Training → Evaluation →
Confusion Matrix → Error Analysis
```

The main goal of this project was to understand:

* how image data is represented,
* how ANN processes images,
* how neural networks learn patterns,
* and how to evaluate deep learning models properly.

---

# 📌 Step 1: Dataset Understanding

MNIST contains handwritten digit images.

### Dataset Size

| Dataset  | Images |
| -------- | ------ |
| Training | 60,000 |
| Testing  | 10,000 |
| Total    | 70,000 |

---

## 🔍 Image Structure

```python
X_train.shape
```

Output:

```python
(60000, 28, 28)
```

Meaning:

* 60,000 training images
* Each image = 28 × 28 pixels

Example:

```python
X_train[0].shape
```

Output:

```python
(28, 28)
```

---

## 🧠 Pixel Values

MNIST images are grayscale.

| Pixel Value | Meaning    |
| ----------- | ---------- |
| 0           | Pure Black |
| 255         | Pure White |
| 120         | Gray       |
| 200         | Light Gray |

### Important Observation

Each image contains:

```python
28 × 28 = 784
```

pixels.

Therefore:

```python
ANN Input Layer = 784 neurons
```

(One neuron per pixel)

---

# 📊 Step 2: Data Visualization

Visualized sample images using Matplotlib.

### Learning

When working with image data, ML engineers usually think:

```python
Load Data
↓
Visualize Data
↓
Check Shape
↓
Check Labels
↓
Normalize Pixels
↓
Prepare for ANN
```

---

# 🧹 Step 3: Data Preprocessing

## 🔹 Pixel Normalization

Converted:

```python
0   → 0.0
255 → 1.0
```

Using:

```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### ✔ Why Normalize?

* Smaller values improve optimization
* Faster convergence
* More stable training

---

## 🔹 Flattening Images

Original Shape:

```python
(60000, 28, 28)
```

Converted To:

```python
(60000, 784)
```

Using:

```python
reshape(60000, 784)
```

### ✔ Why Flatten?

Dense ANN expects:

```python
Feature1
Feature2
Feature3
...
FeatureN
```

A simple list of features.

After flattening:

```python
784 pixels
↓
784 input features
```

---

## 🔹 One-Hot Encoding

Converted labels:

```python
5
```

into:

```python
[0,0,0,0,0,1,0,0,0,0]
```

### ✔ Why?

Softmax produces probabilities for 10 classes.

One-Hot Encoding allows the loss function to compare actual and predicted outputs correctly.

---

# 🧠 Step 4: ANN Architecture

Implemented:

```python
Input Layer (784)

↓

Hidden Layer (128 neurons)
ReLU

↓

Output Layer (10 neurons)
Softmax
```

---

## 🔹 Hidden Layer

```python
Dense(128, activation="relu")
```

### Why ReLU?

* Fast
* Handles non-linearity
* Reduces vanishing gradient issues
* Works very well in deep learning

---

## 🔹 Output Layer

```python
Dense(10, activation="softmax")
```

### Why 10 Neurons?

Digits:

```python
0-9
```

Total:

```python
10 classes
```

### Why Softmax?

Because this is:

```python
Multi-Class Classification
```

Softmax converts outputs into probabilities.

---

# ⚙️ Step 5: Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Learning

| Component                | Purpose                                                  |
| ------------------------ | -------------------------------------------------------- |
| Adam                     | Updates weights and reduces loss                         |
| Categorical Crossentropy | Measures prediction error for multi-class classification |
| Accuracy                 | Tracks model performance                                 |

---

# 🔄 Step 6: Model Training

Training Configuration:

```python
epochs = 5
batch_size = 100
validation_split = 0.2
```

Dataset Split:

| Type       | Images |
| ---------- | ------ |
| Training   | 48,000 |
| Validation | 12,000 |
| Testing    | 10,000 |

---

## 📈 Training Results

| Epoch | Accuracy | Validation Accuracy |
| ----- | -------- | ------------------- |
| 1     | 89.6%    | 93.9%               |
| 2     | 95.0%    | 95.7%               |
| 3     | 96.3%    | 96.2%               |
| 4     | 97.1%    | 96.8%               |
| 5     | 97.8%    | 97.0%               |

### ✔ Observation

Good Learning Pattern:

```python
loss ↓
accuracy ↑

val_loss ↓
val_accuracy ↑
```

Model learned successfully throughout training.

---

# 📊 Step 7: Test Evaluation

Final Results:

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 97.8% |
| Validation Accuracy | 97.0% |
| Test Accuracy       | 97.2% |
| Test Loss           | 0.091 |

---

## 🔹 Overfitting Check

Training and Validation accuracies remained very close:

```python
97.8%
97.0%
97.2%
```

### ✔ Conclusion

No significant overfitting observed.

Model generalized well to unseen data.

---

# 📈 Step 8: Confusion Matrix Analysis

Generated a 10×10 confusion matrix.

### Observation

Most values appeared on the diagonal, meaning:

```python
Actual Digit = Predicted Digit
```

for most images.

Some common mistakes:

* 5 → 6
* 2 → 9
* 4 → 2
* 6 → 0
* 3 → 5

---

# 🔥 Step 9: Misclassified Images Analysis

Total Wrong Predictions:

```python
264
```

out of:

```python
10000
```

test images.

### Important Learning

Most mistakes occurred because of:

* messy handwriting
* unusual digit shapes
* visually similar digits

Examples:

* 5 looked like 6
* 6 looked like 0
* 2 looked like 7
* 3 looked like 5

These were human-like mistakes rather than random errors.

---

# 🧠 Key Learnings

✔ Image data representation

✔ Pixel normalization

✔ Flattening images for ANN

✔ One-Hot Encoding

✔ ANN architecture design

✔ ReLU Activation Function

✔ Softmax Activation Function

✔ Categorical Crossentropy

✔ Adam Optimizer

✔ Training vs Validation vs Testing

✔ Confusion Matrix Interpretation

✔ Error Analysis using Misclassified Images

✔ ANN limitations for image recognition

---

# 🛠 Technologies Used

* Python
* NumPy
* Matplotlib
* TensorFlow
* Keras
* Scikit-learn
* Seaborn

---

# 🎯 Final Conclusion

This project helped me understand the complete Deep Learning workflow from image preprocessing to neural network training and evaluation.

The biggest learning from this project was:

> Deep Learning is not just about building a neural network. Understanding image representation, preprocessing, model architecture, and analyzing mistakes is equally important for building effective AI systems.
