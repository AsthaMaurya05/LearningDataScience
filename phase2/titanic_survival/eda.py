import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
print(df.head())
print(df.info())
print(df.shape)
print(df.describe())
print(df.isnull().sum())


df["Survived"].value_counts()
sns.countplot(x='Survived', data=df)
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df)
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="Survived", y="Age", data = df)
plt.show()

df.drop(columns = ["Cabin"],inplace=True)

df["Age"]=df["Age"].fillna(df["Age"].median())

df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

print(df.isnull().sum())


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

print(df["Embarked"].unique())
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)


X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket"])
y = df["Survived"]
print(X.shape)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("Accuracy (Scaled):", accuracy_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print("AUC:", auc_score)


plt.figure(figsize=(8,5))

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

