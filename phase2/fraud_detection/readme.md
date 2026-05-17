# 💳 Credit Card Fraud Detection using Machine Learning & SMOTE

## 🚀 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning.

The main goal of this project was not just building a model, but understanding:

- Imbalanced datasets
- Why accuracy can fail
- Importance of Recall in fraud detection
- SMOTE (Synthetic Minority Oversampling Technique)
- Precision vs Recall tradeoff
- Threshold Tuning
- Precision-Recall Curve

This project helped me understand how real-world fraud detection systems are designed and evaluated.

---

## 📌 Workflow Followed

**Dataset Understanding** → **EDA** → **Class Imbalance Analysis** → **Baseline Model** → **Evaluation Metrics** → **SMOTE** → **Threshold Tuning** → **Precision-Recall Curve** → **Business Understanding**

---

## 📂 Dataset Information

### Target Column

| Class | Meaning |
|------|------|
| 0 | Genuine Transaction |
| 1 | Fraud Transaction |

### Dataset Shape

```
(284807, 31)
```

### 🔍 Initial Observations

- No missing values present
- Dataset already cleaned
- Fraud transactions were extremely tiny compared to genuine transactions

| Class | Count |
|------|------|
| Genuine | 284315 |
| Fraud | 492 |

### Fraud Percentage

```
(492 / 284807) * 100 ≈ 0.17%
```

**Meaning:** Only **0.17%** transactions were fraud.

This created a **severe imbalanced dataset problem**.

---

## 🧠 Important Industry Learning

In fraud detection: **Recall matters much more than Accuracy**.

**Because:**

| Error Type | Impact |
|------|------|
| False Positive (FP) | Customer inconvenience |
| False Negative (FN) | Money stolen |

---

## 🔀 Train-Test Split

**Used:** `stratify=y`

### Why Stratify?

It preserves the same fraud ratio in:
- Training set
- Testing set

This is **extremely important for imbalanced datasets**.

---

## 🤖 Baseline Logistic Regression Model

First, I trained a baseline Logistic Regression model without handling imbalance.

### 📊 Baseline Results

| Metric | Value |
|------|------|
| Accuracy | 99.91% |

At first glance this looked excellent. But this became an important learning moment: **Accuracy was misleading because almost all transactions were genuine.**

---

## 📈 Fraud Class Metrics

| Metric | Value |
|------|------|
| Precision | 0.83 |
| Recall | 0.63 |
| F1-score | 0.72 |

### Interpretation

- **Precision = 0.83** → Out of all predicted frauds, 83% were actually fraud.
- **Recall = 0.63** → Model detected only 63% frauds.

**Meaning:** Model missed **37% fraud transactions**.

---

## 📊 Baseline Confusion Matrix

```
[[56851    13]
 [   36    62]]
```

| Metric | Meaning |
|------|------|
| TN = 56851 | Correct genuine predictions |
| FP = 13 | False fraud alarms |
| FN = 36 | Fraud missed by model |
| TP = 62 | Correct fraud detection |

### Most Important Observation

**FN = 36** → Actual frauds predicted as genuine.

This is **highly dangerous** because fraud escaped detection.

---

## 🔥 SMOTE (Synthetic Minority Oversampling Technique)

To handle imbalance, I applied **SMOTE**.

**SMOTE creates synthetic fraud samples instead of simply duplicating rows.**

This technique is heavily used in:
- Fraud Detection
- Cybersecurity
- Medical Diagnosis
- Anomaly Detection

---

## 📊 Class Distribution After SMOTE

| Class | Count |
|------|------|
| Genuine | 227451 |
| Fraud | 227451 |

**Dataset became balanced.**

---

## 🤖 Model After SMOTE

### 📈 Results After SMOTE

| Metric | Value |
|------|------|
| Accuracy | 97.4% |

**Accuracy decreased. But fraud detection improved massively.**

---

## 📊 Fraud Metrics After SMOTE

| Metric | Value |
|------|------|
| Precision | 0.06 |
| Recall | 0.92 |

---

## 📊 Confusion Matrix After SMOTE

```
[[55397 1467]
 [    8   90]]
```

| Metric | Meaning |
|------|------|
| TN = 55397 | Correct genuine predictions |
| FP = 1467 | False fraud alarms |
| FN = 8 | Fraud missed |
| TP = 90 | Correct fraud detection |

### 🔥 Biggest Success

**False Negatives reduced:**

| Before | After |
|------|------|
| 36 | 8 |

**Very few frauds escaped detection after SMOTE.**

---

## ⚖️ Precision vs Recall Tradeoff

This project clearly demonstrated: **Improving Recall often sacrifices Precision.**

After SMOTE:
- **Recall improved massively**
- **But Precision collapsed**

**Meaning:**
- Model became better at catching frauds
- But also produced many false alarms

---

## 🎯 Threshold Tuning

Instead of using default threshold: **0.5**

I manually tuned thresholds to control:
- Fraud sensitivity
- Precision
- Recall

### 📊 Threshold Comparison

| Model | Accuracy | Fraud Recall | False Positives |
|------|------|------|------|
| Baseline | 99.9% | 0.63 | 13 |
| SMOTE | 97% | 0.92 | 1467 |
| Threshold = 0.8 | 99% | 0.90 | 544 |
| Threshold = 0.3 | 94% | 0.92 | 3175 |

---

## 🧠 Threshold Learning

### Lower Threshold

**Effects:**
- Higher Recall
- More fraud detection
- More false alarms
- Lower Precision

### Higher Threshold

**Effects:**
- Better Precision
- Fewer false alarms
- Slightly lower Recall
- Misses slightly more frauds

---

## 📈 Precision-Recall Curve

Since fraud detection is highly imbalanced, I also analyzed:

- **Precision-Recall Curve**
- **Average Precision (AP) Score:** 0.72

This showed: **Model maintained reasonably strong fraud detection capability across thresholds.**

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 🎯 Final Conclusion

This project completely changed my understanding of classification problems.

**The biggest learning was:** In imbalanced datasets, high accuracy does not necessarily mean a good model.

**A model with lower accuracy but higher fraud Recall can be far more valuable in real-world fraud detection systems.**

This project helped me understand how Machine Learning decisions are connected to real business risks and tradeoffs.