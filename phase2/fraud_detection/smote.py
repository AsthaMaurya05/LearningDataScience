import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled,y_train)


print("\nBEFORE SMOTE")
print(y_train.value_counts())

print("\nAFTER SMOTE")
print(pd.Series(y_train_smote).value_counts())


plt.figure(figsize=(6,4))
sns.countplot(x=y_train_smote)
plt.title("Balanced Classes After SMOTE")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

model_smote = LogisticRegression()
model_smote.fit(X_train_smote,y_train_smote)
y_pred_smote = model_smote.predict(X_test_scaled)


print("\nSMOTE MODEL RESULT")
print("Accuracy:",accuracy_score(y_test, y_pred_smote))

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred_smote))

cm = confusion_matrix(y_test,y_pred_smote)
print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SMOTE")
plt.show()