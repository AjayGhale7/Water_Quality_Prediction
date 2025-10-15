# train_model_xgb_smote.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv(r"d:\Dashboard\water_potability.csv")

# 2. Handle missing values
df = df.fillna(df.median())

# 3. Features (X) and Target (y)
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Apply SMOTE on training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", X_train.shape, y_train.value_counts().to_dict())
print("After SMOTE:", X_train_res.shape, pd.Series(y_train_res).value_counts().to_dict())

# 6. Build XGBoost model
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train_res, y_train_res)

# 7. Predictions
y_pred = xgb.predict(X_test)

# 8. Evaluation
print("✅ XGBoost + SMOTE Model Training Complete")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Save model
joblib.dump(xgb, "water_quality_xgb_smote.pkl")
print("🎉 Model saved as water_quality_xgb_smote.pkl")
