# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load dataset
df = pd.read_csv(r"d:\Dashboard\water_potability.csv")  # update path if needed

# 2. Handle missing values (fill with mean)
df = df.fillna(df.mean())

# 3. Features (X) and Target (y)
X = df.drop("Potability", axis=1)   # all features
y = df["Potability"]                # target column

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Build Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Predictions
y_pred = rf.predict(X_test)

# 7. Evaluation
print("✅ Model Training Complete")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Save model
joblib.dump(rf, "water_quality_model.pkl")
print("🎉 Model saved as water_quality_model.pkl")
