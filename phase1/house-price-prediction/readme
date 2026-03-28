# 🏠 House Price Prediction (My ML Learning Project)

## 📌 What I Did in This Project

This project is part of my ML learning journey.
My goal was not just to build a model, but to **understand how to think like a data scientist step by step**.

---

# 🚀 Step 1: Starting with Data

I first started by loading the dataset and checking basic things:

* `df.head()`
* `df.shape`
* `df.info()`

From this, I understood:

* what columns are there
* which are numerical and which are categorical
* there are no missing values

---

# 📊 Step 2: Understanding the Target (Price)

Then I plotted the histogram of `price`.

👉 I observed that:

* price is **right-skewed**

So I applied:

```python
df['price'] = np.log(df['price'])
```

👉 This helped make the data more normal for modeling.

---

# 📈 Step 3: Understanding Numerical Features

Then I analyzed numerical columns like:

* area, bedrooms, bathrooms, stories, parking

I used:

* histograms → to see distribution
* scatter plots → to see relation with price

---

### 🔍 What I Observed:

* `area` has strong impact on price 🔥
* `bathrooms` also affects price
* `bedrooms`, `stories`, `parking` are less strong

👉 From this I learned:

> Just seeing numbers is not enough, visualization is important

---

# 🔗 Step 4: Correlation

Then I created a correlation heatmap.

👉 I observed:

* area → highest correlation
* bathrooms → strong
* others → moderate

👉 This confirmed what I saw in scatter plots.

---

# 📦 Step 5: Understanding Categorical Features

Then I analyzed categorical features using **boxplots**.

---

### 🔍 Observations:

* `airconditioning`, `mainroad`, `prefarea` → strong impact
* `basement` → medium
* `guestroom`, `hotwaterheating` → weak

👉 I learned:

> Don’t remove features blindly — first check their impact

---

# ⚙️ Step 6: Data Preprocessing

After understanding the data, I did preprocessing:

### ✔ Converted:

* yes → 1
* no → 0

### ✔ Applied:

* One-hot encoding for `furnishingstatus`

---

# 🤖 Step 7: Model Building (First Attempt)

Then I built my first model using **Linear Regression**.

---

### 📊 Results:

* R² ≈ **0.67**

👉 This means:

> My model explains about 67% of the price variation

---

# 🧠 Step 8: Understanding Model Output

I checked coefficients and found:

* bathrooms, AC, prefarea → strong features
* bedrooms → weak
* unfurnished → negative impact

👉 This matched my EDA insights ✔️

---

# 🚀 Step 9: Improving the Model

After this, I created a **new file (model_training.py)** to improve the model.

In that file I:

* applied log transformation
* encoded data
* did train-test split
* applied scaling
* trained model again

---

# 🌳 Step 10: Tried Random Forest

I also tried a more complex model:

👉 Random Forest

---

### 📊 Result:

* R² ≈ 0.63 (lower than linear regression)

---

# 🧠 Final Learning (MOST IMPORTANT)

```text
Complex model is not always better
Simple model can perform better on small data
```

---

# 🎯 Final Decision

* Linear Regression performed better
* So I decided to keep it as my final model

---

# 🔥 What I Learned from This Project

* How to do proper EDA
* How to read graphs (histogram, scatter, boxplot)
* Difference between correlation and actual relationship
* Importance of preprocessing
* How to build and evaluate models
* How to compare models
* How to think step-by-step like a data scientist

---

# 🚀 What’s Next

Now I will move to next project:

👉 **Iris Flower Classification**

to learn:

* classification problems
* logistic regression
* KNN

---

# 💬 Final Thought

Before this project:

```text
I was just coding ML
```

After this project:

```text
Now I understand how ML actually works step by step
```

---

👨‍💻 *This project is part of my journey to become an AI & Data Science Engineer*
