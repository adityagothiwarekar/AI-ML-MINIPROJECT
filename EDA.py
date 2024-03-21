import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check columns and handle missing values
print(train.columns)
print(test.columns)
print(train.isna().sum())

# Handle missing values by forward filling
train = train.ffill()
test = test.ffill()

# Drop 'Id' column from train dataset
train = train.drop("Id", axis=1)

# Separate categorical and numerical columns
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=np.number).columns.tolist()

# Encode categorical features using One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train[categorical_cols]))
train_encoded.columns = encoder.get_feature_names(categorical_cols)

# Replace original categorical columns with encoded ones
train = pd.concat([train.drop(categorical_cols, axis=1), train_encoded], axis=1)

# Scale numerical features using StandardScaler
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])

# Split train dataset into features and target variable
X = train.drop(columns='SalePrice')
y = train['SalePrice']

# Split train dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest Regression model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Initialize and train Support Vector Regression (SVR) model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Initialize and train Gradient Boosting Regression model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Initialize and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on training and testing sets for each model
models = [rf_model, svr_model, gb_model, lr_model,dt_model]
model_names = ['Random Forest', 'Support Vector', 'Gradient Boosting', 'Linear Regression','Decision Tree']
''
for model, name in zip(models, model_names):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n{name}:")
    print("Training RMSE:", rmse_train)
    print("Test RMSE:", rmse_test)
    print("Training R-squared score:", model.score(X_train, y_train))
    print("Test R-squared score:", model.score(X_test, y_test))


