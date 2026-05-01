# 📊 Customer Churn Prediction (Decision Tree & Random Forest)

## 🚀 My Approach (Step-by-Step Learning Flow)

This project is part of my ML learning journey where I am focusing on understanding how models actually work, not just applying them.

---

# 📌 Step 1: Data Understanding (EDA)

I first explored the dataset.

✔ No missing values in most columns
✔ Target column → **Churn Label (Yes / No)**

---

## 📊 Target Distribution

I plotted countplot and observed:

* Yes ≈ 1800 (~25%)
* No ≈ 5000 (~75%)

### 🧠 Observation:

* Most customers are not churning
* Dataset is **imbalanced**

👉 Important learning:

```text
If dataset is imbalanced → model may always predict "No"
```

---

## ⚠️ Key Insight

```text
Accuracy alone is not enough
We need precision, recall, confusion matrix
```

---

# 📊 Step 2: Feature Analysis (Boxplots)

I analyzed numerical features:

### ✔ Observations:

```text
1. High Monthly Charges → churn increases
2. Low Tenure → churn increases
3. Low Total Charges → churn increases
```

👉 Meaning:

* New + expensive customers are more likely to leave

---

# 🧠 Step 3: Understanding Metrics

I learned:

* **Precision** → How correct are predicted churns
* **Recall** → How many actual churn customers we found
* **F1-score** → balance of precision & recall
* **Support** → number of samples
* **Accuracy** → overall correctness (not class-specific)

---

# 🌳 Step 4: Decision Tree

👉 Decision Tree = **one brain (rule-based model)**

### Result:

```text
Accuracy ≈ 0.80
Churn Recall ≈ 0.61
```

✔ Model is decent
❌ But missing many churn customers

---

# 🌲 Step 5: Random Forest (First Attempt)

👉 Random Forest = **many trees voting together**

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

### Result:

```text
Accuracy ≈ 0.71
Churn Recall = 0.00 ❌
```

---

## 🧠 Why Random Forest Failed?

* Dataset is imbalanced
* Model became conservative
* Predicted mostly "No"

### Key Learning:

```text
Better model ≠ better performance
Random Forest fails on imbalanced data without tuning
```

---

# 🌲 Step 6: Random Forest (Fixed)

I fixed imbalance using:

```python
class_weight='balanced'
```

### Result:

```text
Accuracy ↓
Recall ↑ (very important)
```

---

## 🧠 Business Thinking

```text
Miss churn customer → very bad
Wrongly warn non-churn → acceptable
```

👉 So recall is more important than accuracy

✔ Final model → **Random Forest**

---

# 📊 Step 7: Feature Importance

Top features:

```text
- Payment Method
- Contract
- Internet Service
- Security / Support
- Tenure Months
```

👉 These features influence churn most

---

# 🎯 Step 8: Threshold Tuning (Most Important Part)

Instead of default threshold = 0.5, I tried:

| Threshold | Effect                     |
| --------- | -------------------------- |
| 0.3       | High recall, low precision |
| 0.5       | Balanced                   |
| 0.7       | High precision, low recall |

---

## 🧠 Key Insight

```text
Recall ↑ → catch more churn
Precision ↓ → more false alarms
```

---

## ❓ Which Threshold is Best?

```text
There is NO single best threshold
It depends on business goal
```

---

# 🧠 FINAL LEARNINGS

```text
1. Data imbalance is critical in real-world problems
2. Accuracy alone is misleading
3. Recall is more important in churn prediction
4. Decision Tree is simple but less stable
5. Random Forest is powerful but needs tuning
6. Feature importance helps understand model behavior
7. Threshold tuning controls precision vs recall
8. ML is not just coding — it is decision making based on business needs
```

---

# 🎯 CONCLUSION

This project helped me understand:

* How to approach real-world datasets
* How models behave differently on imbalanced data
* How to think beyond accuracy
* How to tune models based on business requirements

---


## 💡 Final Thought

This project was not about getting highest accuracy.
It was about understanding **how and why models behave the way they do**.

---
