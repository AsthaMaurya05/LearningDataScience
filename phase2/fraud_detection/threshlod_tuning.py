import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,confusion_matrix)

df = pd.read_csv("creditcard.csv")


X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled,y_train)

model = LogisticRegression()
model.fit(X_train_smote,y_train_smote)


y_prob = model.predict_proba(X_test_scaled)[:, 1]

threshold = 0.3
y_pred_custom = (y_prob >= threshold).astype(int)

print(f"\nTHRESHOLD = {threshold}")
print("\nClassification Report:\n")
print(classification_report(y_test,y_pred_custom))

cm = confusion_matrix(y_test,y_pred_custom)
print("\nConfusion Matrix:\n")
print(cm)


plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title(f"Threshold = {threshold}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#PR CURVE AND AVERAGE PRECISION SCORE
from sklearn.metrics import precision_recall_curve,average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test,y_prob)

ap_score = average_precision_score(
    y_test,
    y_prob
)

print("Average Precision Score:", ap_score)

plt.figure(figsize=(8,5))
plt.plot(
    recall,
    precision,
    label=f"AP Score = {ap_score:.2f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()