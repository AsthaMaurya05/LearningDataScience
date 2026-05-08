
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("train.csv")
df.drop(columns=["Cabin"], inplace=True)

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0,"female": 1})

df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)


#  FEATURE ENGINEERING

df["FamilySize"] = (df["SibSp"] + df["Parch"] + 1)

print(df[["SibSp", "Parch", "FamilySize"]].head())

df["IsAlone"] = 0

df.loc[df["FamilySize"] == 1,"IsAlone"] = 1

print(df[["FamilySize", "IsAlone"]].head(10))



# FEATURE ANALYSIS

sns.countplot(x="FamilySize",hue="Survived",data=df)
plt.title("Family Size vs Survival")
plt.show()


sns.countplot(x="IsAlone",hue="Survived",data=df)
plt.title("IsAlone vs Survival")
plt.show()



# TITLE EXTRACTION

df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.",expand=False)

print(df[["Name", "Title"]].head(10))
print(df["Title"].value_counts())


df["Title"] = df["Title"].replace(
    [
        'Lady', 'Countess', 'Capt',
        'Col', 'Don', 'Dr',
        'Major', 'Rev', 'Sir',
        'Jonkheer', 'Dona'
    ],
    'Rare'
)

df["Title"] = df["Title"].replace('Mlle','Miss')

df["Title"] = df["Title"].replace('Ms','Miss')

df["Title"] = df["Title"].replace('Mme','Mrs')

print(df["Title"].value_counts())



sns.countplot(x="Title",hue="Survived",data=df)
plt.xticks(rotation=45)
plt.title("Title vs Survival")
plt.show()


df = pd.get_dummies(df,columns=["Title"],drop_first=True)
print(df.head())

X = df.drop(columns=["Survived","PassengerId","Name","Ticket"])
y = df["Survived"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score)

new_accuracy = accuracy_score(y_test,y_pred)
print(new_accuracy)
print(classification_report(y_test, y_pred))

#Comparision

print("Baseline Model Accuracy ≈ 0.81")
print(f"Feature Engineered Accuracy = {new_accuracy}")