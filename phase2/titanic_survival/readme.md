# 🚢 Titanic Survival Prediction (Logistic Regression & Feature Engineering)

## 🚀 Project Overview

This project focuses on predicting whether a passenger survived the Titanic disaster using Machine Learning.

Instead of directly training a model, I followed a proper ML workflow:

```python
Data Understanding → EDA → Cleaning → Encoding →
Baseline Model → Evaluation → Feature Engineering → Improvement
```

The main goal of this project was to understand:

* how real-world datasets are analyzed,
* how preprocessing decisions are made,
* and how feature engineering improves model performance.

---

# 📌 Step 1: Dataset Understanding

I loaded the Titanic dataset and checked:

* dataset structure,
* missing values,
* datatypes,
* and feature importance.

## 🔍 Missing Values Found

| Column   | Missing Values |
| -------- | -------------- |
| Age      | 177            |
| Cabin    | 687            |
| Embarked | 2              |

---

## 🧠 Initial Observations

Categorical columns:

* Name
* Sex
* Ticket
* Cabin
* Embarked

Important possible survival factors:

* Sex
* Age
* Pclass

---

# 📊 Step 2: Exploratory Data Analysis (EDA)

## 🔹 Key Observations

### Survival Distribution

* Deaths were much higher than survivors
* Dataset was slightly imbalanced

### Gender vs Survival

* Females had much higher survival rates
* Most males did not survive

### Passenger Class vs Survival

* Third-class passengers had the highest death rate
* First-class passengers survived more

### Age Distribution

* Most passengers were between 15–40 years old
* Younger passengers showed slightly better survival chances

---

# 🧹 Step 3: Handling Missing Values

| Column   | Action             | Reason                  |
| -------- | ------------------ | ----------------------- |
| Cabin    | Dropped            | Too many missing values |
| Embarked | Filled with Mode   | Categorical feature     |
| Age      | Filled with Median | Presence of outliers    |

---

# 🔤 Step 4: Feature Encoding

## 🔹 Sex Encoding

| Category | Encoded |
| -------- | ------- |
| Male     | 0       |
| Female   | 1       |

Used:

```python
map()
```

---

## 🔹 Embarked Encoding

Applied One-Hot Encoding:

| Embarked | Embarked_C | Embarked_Q |
| -------- | ---------- | ---------- |
| S        | 0          | 0          |
| C        | 1          | 0          |
| Q        | 0          | 1          |

### 🧠 Encoding Learning

| Situation                         | Best Encoding    |
| --------------------------------- | ---------------- |
| Two categories                    | map()            |
| Multiple categories without order | One-Hot Encoding |

---

# 🧠 Step 5: Feature Selection

Selected important features:

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked columns

Removed:

* PassengerId
* Ticket
* Name

---

# 🔀 Step 6: Train-Test Split & Scaling

* 80% training
* 20% testing

Applied:

```python
StandardScaler()
```

### ✔ Why Scaling?

Features had different ranges:

* Sex → 0–1
* Fare → 0–500+

Scaling improves optimization and convergence.

---

# 🤖 Step 7: Baseline Model

Implemented:

```python
LogisticRegression()
```

## 📈 Baseline Accuracy

```python
0.81
```

---

# 📊 Step 8: Confusion Matrix & Classification Metrics

Confusion Matrix:

```python
[[90 15]
 [19 55]]
```

## 🔹 Important Observation

Recall:

* Dead Class → 0.86
* Survivor Class → 0.74

The model was better at identifying deaths than survivors.

---

# 🔥 Step 9: ROC Curve & AUC Score

## 🔹 AUC Score

```python
0.88
```

### ✔ Interpretation

* ROC curve bent toward the top-left corner
* High TPR and low FPR
* Model separated survivors and non-survivors effectively

---

# 🧠 Step 10: Feature Engineering (MOST IMPORTANT PART)

## 🔹 FamilySize Feature

Created:

```python
FamilySize = SibSp + Parch + 1
```

### ✔ Observation

* Family sizes 2–4 survived relatively better
* Alone passengers survived less

---

## 🔹 IsAlone Feature

Created:

```python
IsAlone
```

to identify passengers traveling alone.

---

## 🔹 Title Extraction from Names

Extracted titles like:

* Mr
* Mrs
* Miss
* Master

Example:

```python
Braund, Mr. Owen Harris → Mr
```

### ✔ Learning

Titles indirectly contain:

* gender,
* age,
* social role,
* social status.

Rare titles were grouped into:

```python
Rare
```

---

# 📈 Step 11: Model Comparison

| Model                    | Accuracy |
| ------------------------ | -------- |
| Baseline Model           | 0.81     |
| Feature Engineered Model | 0.815    |

## 🔹 Important Improvement

Recall for Survivor Class improved:

| Before | After |
| ------ | ----- |
| 0.74   | 0.77  |

The model became better at identifying actual survivors after feature engineering.

---

# 🧠 Key Learnings

✔ Importance of EDA
✔ Handling missing values properly
✔ Difference between Mean, Median & Mode
✔ One-Hot Encoding
✔ Logistic Regression workflow
✔ Confusion Matrix interpretation
✔ ROC Curve & AUC understanding
✔ Feature Engineering using real-world intuition
✔ Better features often improve models more than changing algorithms

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# 🎯 Final Conclusion

This project helped me understand the complete Machine Learning workflow step-by-step.

The biggest learning from this project was:

> Good Machine Learning depends heavily on data understanding, preprocessing, and feature engineering.
