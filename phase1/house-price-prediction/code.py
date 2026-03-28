import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("housing.csv")

print(df.head())
print(df.shape)
print(df.info())



plt.hist(df['price'])
plt.title("Price Distribution")
plt.show()   
#Is price normally distributed? Or skewed?

df['price'] = np.log(df['price'])   #because the data is skewed, we can apply log transformation to make it more normal

# Understand EACH Feature (Univariate Analysis)
num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

for col in num_cols:
    plt.hist(df[col])
    plt.title(col)
    plt.show()
    
    

#Relationship with Target (PRICE)
num_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

for col in num_cols:
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f"{col} vs Price")
    plt.show()
    


#preprocessing
#covert yes ->1 and no ->0 for binary columns
binary_cols = ['mainroad', 'guestroom', 'basement',
               'hotwaterheating', 'airconditioning', 'prefarea']

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

#. Convert furnishingstatus    
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)   


#correlation matrix
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)  
plt.title("Correlation Matrix")
plt.show()       
    
sns.boxplot(x='airconditioning', y='price', data=df)
plt.show()    

# Airconditioning has a strong positive impact on price.
# Houses with AC are generally more expensive.
# However, overlap indicates other features also influence price.

sns.boxplot(x='mainroad', y='price', data=df)
plt.show() 

sns.boxplot(x='guestroom', y='price', data=df)
plt.show() 

sns.boxplot(x='basement', y='price', data=df)
plt.show() 

sns.boxplot(x='hotwaterheating', y='price', data=df)
plt.show() 

sns.boxplot(x='prefarea', y='price', data=df)
plt.show() 
    
# Feature	Importance
# mainroad	Strong
# prefarea	Strong
# basement	Medium
# guestroom	Low
# hotwaterheating	Weak

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)


# right now r2 = 0.67 which means our model explains 67% of the variance in the target variable (price).
#coefficients indicate the strength and direction of the relationship between each feature and the target variable. 
# For example, a positive coefficient for 'area' suggests that as the area increases, the price tends to increase, while a negative coefficient for 'bedrooms' would suggest that more bedrooms are associated with lower prices, all else being equal.

#bathrooms = 0.186, airconditioning = 0.157, prefarea = 0.125, mainroad = 0.111, basement = 0.099  - these  are top features that have strong positive impact on price.
#stories = 0.082, parking = 0.053, guestroom = 0.036,  - these are medium features.
#bedrooms = 0.020 - low imapct on price, unfurnished = -0.109 -negative impact

#to improve the model we can ................
# 1.feature scaling
# 2.remove weak features
# 3.try better model




