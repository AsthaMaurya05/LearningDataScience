
# 🌸 Iris Flower Classification (Logistic Regression & KNN)

## 🚀 My Learning Approach (Step-by-Step Thinking)

This project is part of my ML revision journey where I’m focusing on **understanding concepts deeply**, not just applying models.

Instead of directly building a model, I followed a **proper data science workflow**.

---

# 📌 Step 1: Loading the Data

I started by loading the Iris dataset using sklearn.

I converted it into a pandas DataFrame and checked:

* `df.head()` → to see how data looks
* `df.info()` → to check datatypes and missing values

✔ Observations:

* Dataset has 150 rows and 5 columns
* All features are numerical
* No missing values
* Target has 3 classes → 0, 1, 2

---

# 📊 Step 2: Initial Data Understanding

## 🔹 Target Distribution

I plotted a countplot:

👉 All classes have equal data (balanced dataset)

✔ Learning:

* No class imbalance → good for model training

---

## 🔹 Univariate Analysis (Histograms)

I plotted histograms for all features.

✔ Observations:

* Features are mostly normally distributed
* No major outliers

✔ Learning:

* Data is clean → no heavy preprocessing needed

---

## 🔥 Step 3: Relationship Between Features (Pairplot)

I used:

```python
sns.pairplot(df, hue='target')
```

✔ Key Observations:

* 🌟 **Petal length & petal width clearly separate classes**
* Sepal features show overlap (especially class 1 & 2)

✔ My Conclusion:

* Petal features are **most important**
* Sepal width is **weak feature**

---

## 🎨 Feature Pattern Analysis

### 🔹 Univariate Feature Patterns:

| Feature | Pattern | Observation |
|---------|---------|-------------|
| **Sepal Length** | Somewhat normal (bell-like) | Distributed across classes |
| **Sepal Width** | Slight variation | Not very wide distribution |
| **Petal Length** | Clear separation into groups | **STRONG** - distinct for each class |
| **Petal Width** | Multiple clusters | **STRONG** - separates classes well |

### 🔹 Target Distribution:
* 3 balanced bars for classes (0, 1, 2)
* Equal representation of all classes

### 🔹 Feature Importance Classification:

#### 💪 **STRONG FEATURES** (Best for Classification):
* **Petal Length** - Clear group separation
* **Petal Width** - Multiple distinct clusters

✔ Why: Features that show clear separation between classes are best for classification

#### 🤔 **WEAK FEATURES**:
* **Sepal Length** - Limited discriminative power
* **Sepal Width** - Significant overlap between classes

✔ Why: Sepal measurements don't clearly distinguish between classes

### 🧠 Key Insight:

👉 **Features with clear separation → Better model performance**

---

# 🧠 Step 4: Feature & Target Selection

```python
X = df.drop('target', axis=1)
y = df['target']
```

---

# 🔀 Step 5: Train-Test Split

```python
train → learning
test → evaluation
```

✔ Used:

* 80% training
* 20% testing

---

# ⚖️ Step 6: Feature Scaling

I applied StandardScaler.

✔ Why?

* KNN is distance-based → scale matters
* Logistic regression also performs better

---

# 🤖 Step 7: Model Training

## 🔹 Logistic Regression

```python
lr = LogisticRegression()
```

## 🔹 KNN

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

---

# 📈 Step 8: Model Evaluation

## 🔹 Results:

* Logistic Regression Accuracy = **1.0**
* KNN Accuracy = **1.0**

---

## 🧠 Important Realization

Even though accuracy is perfect:

👉 This dataset is **very easy**

✔ Learning:

* High accuracy ≠ powerful model
* Sometimes data itself is simple

---

# 📊 Step 9: Confusion Matrix (VERY IMPORTANT)

I plotted confusion matrix for Logistic Regression.

✔ Result:

```
[10 0 0]
[0  9 0]
[0  0 11]
```

✔ Detailed Class-wise Analysis:

## 🔹 CLASS 0: 
```
True = 0 → [10, 0, 0]
✅ 10 correctly predicted as 0
❌ 0 wrongly predicted as 1
❌ 0 wrongly predicted as 2
✔ PERFECT
```

## 🔹 CLASS 1:
```
True = 1 → [0, 9, 0]
✅ 9 correctly predicted as 1
❌ 0 misclassifications
✔ PERFECT
```

## 🔹 CLASS 2:
```
True = 2 → [0, 0, 11]
✅ 11 correctly predicted as 2
❌ 0 misclassifications
✔ PERFECT
```

✔ Interpretation:

* All predictions are correct across all classes
* No misclassification in any class
* Model achieved perfect classification

---

## 🧠 Key Learning

* Accuracy gives summary
* Confusion matrix gives detailed truth

---

# 🧪 Step 10: Experiments (MOST IMPORTANT PART)

This is where I actually learned deeply.

---

## 🔹 Experiment:  Remove Important Features

I removed petal features:

```python
X = df[['sepal length (cm)', 'sepal width (cm)']]
```

✔ Result:

* Accuracy dropped - logistic - 0.9, knn - 0.7666666666666667
* Classes started overlapping - Class 0 → still mostly correct, Class 1 & 2 → confusion appears

✔ Learning:
👉 Good features are more important than models


# 🧠 FINAL LEARNINGS

✔ Always start with EDA
✔ Understand features before modeling
✔ Visualization is more powerful than numbers
✔ Good features → better model
✔ Accuracy alone is not enough
✔ Confusion matrix helps understand errors
✔ Simple models can perform very well on clean data

---

# 🎯 Conclusion

This project helped me understand:

* How to approach a dataset step-by-step
* How to analyze features before modeling
* How models behave with different inputs
---

# 📸 Visualizations & Analysis Plots

## 🔹 Exploratory Data Analysis (EDA)

![Countplot - Target Distribution](images/Screenshot%202026-04-26%20220650.png)
*Target distribution showing balanced dataset*

![Histograms - Feature Distributions](images/Screenshot%202026-04-26%20220706.png)
*Univariate analysis of features - Normal distributions*

![Pairplot - Feature Relationships](images/Screenshot%202026-04-26%20222414.png)
*Pairplot showing relationships between features and class separation*

## 🔹 Model Evaluation

![Confusion Matrix - Logistic Regression](images/Screenshot%202026-04-26%20225453.png)
*Perfect classification - All predictions correct*

![Feature Importance Analysis](images/Screenshot%202026-04-26%20225741.png)
*Visual representation of feature importance*

## 🔹 Additional Insights

![Dataset Visualization 1](images/WhatsApp%20Image%202026-04-26%20at%2011.22.00%20PM.jpeg)
*Detailed analysis visualization*

![Dataset Visualization 2](images/WhatsApp%20Image%202026-04-26%20at%2011.22.01%20PM.jpeg)
*Supplementary analysis visualization*
---

# 🚀 What’s Next?

Moving to next project in my learning roadmap:

👉 **Customer Churn Prediction (Decision Trees & Random Forest)**

---
