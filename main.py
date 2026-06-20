import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import xgboost
import shap

print("Everything working perfectly 🚀")

import pandas as pd

# Load dataset
df = pd.read_excel("final_credit_dataset.xlsx")

# Show first 5 rows
print(df.head())

# Show basic info
print("\nDataset Info:\n")
print(df.info())

# Show column names
print("\nColumns:\n", df.columns)

# Check for missing values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Basic statistics
print("\nStatistical Summary:\n")
print(df.describe())

# Check unique values in each column (important for categorical data)
print("\nUnique Values per Column:\n")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")


print("\n--- Data Preprocessing ---\n")

# 1. Drop unnecessary column
df = df.drop("id", axis=1)

# 2. Handle missing values (fill with mean)
df["fea_2"].fillna(df["fea_2"].mean(), inplace=True)

# 3. Separate features and target
X = df.drop("label", axis=1)
y = df["label"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 4. Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape) 


print("\n--- Model Training (XGBoost) ---\n")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Create model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# 2. Train model
model.fit(X_train, y_train)

print("Model trained successfully ✅")

# 3. Predictions
y_pred = model.predict(X_test)

# 4. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))



print("\n--- SHAP Explanation ---\n")

import shap

# 1. Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# 2. Calculate SHAP values
shap_values = explainer(X_test)

print("SHAP values calculated ✅")

# 3. Global Feature Importance (Summary Plot)
shap.summary_plot(shap_values, X_test)

# 4. Individual Prediction Explanation (first sample)
shap.plots.waterfall(shap_values[0])



print("\n--- SHAP Explanation ---\n")

import shap

# 1. Create SHAP explainer
explainer = shap.Explainer(model, X_train)

# 2. Calculate SHAP values
shap_values = explainer(X_test)

print("SHAP values calculated ✅")

# 3. Global Feature Importance (Summary Plot)
shap.summary_plot(shap_values, X_test)

# 4. Individual Prediction Explanation (first sample)
shap.plots.waterfall(shap_values[0])



print("\n--- Saving Model ---\n")

import joblib

# Save model
joblib.dump(model, "final_credit_dataset_model.pkl")

print("Model saved as credit_risk_model.pkl ✅")



# Load model later
loaded_model = joblib.load("credit_risk_model.pkl")



print("\n--- Sample Prediction Report ---\n")

# Take one sample
sample = X_test.iloc[0:1]

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction (0 = Low Risk, 1 = High Risk):", prediction[0])
print("Probability:", probability)



print("\n--- Generating Graphs ---\n")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Target Distribution (Label count)
plt.figure()
sns.countplot(x=y)
plt.title("Target Distribution (Credit Risk)")
plt.xlabel("Risk (0 = Low, 1 = High)")
plt.ylabel("Count")
plt.savefig("target_distribution.png")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# 3. Feature Importance (from XGBoost)
plt.figure(figsize=(10,6))
import pandas as pd

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.show()

