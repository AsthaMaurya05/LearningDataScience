import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('customer_churn.csv')

print(df.head())
print(df.info())
print(df.shape)



# # Target distribution
# sns.countplot(x='Churn Label', data=df)
# plt.title("Churn Distribution")
# plt.show()

# # Monthly Charges vs Churn
# sns.boxplot(x='Churn Label', y='Monthly Charges', data=df)
# plt.title("Monthly Charges vs Churn")
# plt.show()

# # Tenure vs Churn
# sns.boxplot(x='Churn Label', y='Tenure Months', data=df)
# plt.title("Tenure vs Churn")
# plt.show()


# Total Charges was string → convert to numeric
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(0)


# # Total Charges vs Churn
# sns.boxplot(x='Churn Label', y='Total Charges', data=df)
# plt.title("Total Charges vs Churn")
# plt.show()


# Remove leakage + useless columns
df = df.drop([
    'CustomerID',
    'Churn Value',
    'Churn Score',
    'CLTV',
    'Churn Reason',

    'Zip Code',
    'Country',
    'State',
    'City',
    'Lat Long',
    'Count'
], axis=1)


# Convert target to numeric
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

# Convert all categorical columns
df = pd.get_dummies(df, drop_first=True)


from sklearn.model_selection import train_test_split

X = df.drop('Churn Label', axis=1)
y = df['Churn Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop Important Features:\n")
print(feature_importance.sort_values(ascending=False).head(10))


# Check training accuracy
train_pred = model.predict(X_train)

print("\nTrain Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Report:\n")
print(classification_report(y_test, rf_pred))


rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)

print("\nRandom Forest Important Features:\n")
print(rf_importance.sort_values(ascending=False).head(10))


rf_train_pred = rf_model.predict(X_train)

print("\nRF Train Accuracy:", accuracy_score(y_train, rf_train_pred))
print("RF Test Accuracy:", accuracy_score(y_test, rf_pred))

y_prob = rf_model.predict_proba(X_test)[:, 1] #gives Probability of churn for each customer
y_pred_custom = (y_prob > 0.6).astype(int)
print("\nCustom Threshold (0.6) Report:\n")
print(classification_report(y_test, y_pred_custom))
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob > t).astype(int)
    print(f"\nThreshold: {t}")
    print(classification_report(y_test, y_pred_t))