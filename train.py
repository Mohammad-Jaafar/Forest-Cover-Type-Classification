# =============================
# Forest Cover Type Classification
# =============================

# Import libraries
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------
data = fetch_covtype(as_frame=True)
df = data.frame

# ---------------------------------------------
## 2. Explore the data
# ---------------------------------------------
print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())

# ---------------------------------------------
## 3. Separate features and target
# ---------------------------------------------
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# ---------------------------------------------
# 4. Split data into train and test
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# 5. Train Random Forest
# ---------------------------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# ---------------------------------------------
# 6. Train XGBoost
# ---------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

y_xgb_train = y_train - 1
y_xgb_test = y_test - 1

xgb_model.fit(X_train, y_xgb_train)
xgb_preds = xgb_model.predict(X_test)

xgb_preds += 1

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

# ---------------------------------------------
# 7. Confusion Matrix
# ---------------------------------------------
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------
# 8. Feature Importance
# ---------------------------------------------
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features (XGBoost)")
plt.show()

# ---------------------------------------------
# 9. Compare Models
# ---------------------------------------------
rf_acc = accuracy_score(y_test, rf_preds)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("Model Comparison:")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

if xgb_acc > rf_acc:
    print("✅ XGBoost performed better!")
else:
    print("✅ Random Forest performed better!")
