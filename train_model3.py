# =====================================================
# 🚰 Water Potability Prediction - Improved Stacking Ensemble
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------------
# 1. Load Balanced Dataset
# -------------------------------
df = pd.read_csv("D:/Dashboard/water_potability_balanced.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 2. Define Base Models
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    gamma=0,
    random_state=42,
    eval_metric="logloss"
)

log_reg = LogisticRegression(max_iter=1000, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)

# -------------------------------
# 3. Stacking Ensemble
#    (Meta-learner = LogisticRegression, passthrough=True)
# -------------------------------
stack_model = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("logreg", log_reg), ("gb", gb)],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1,
    passthrough=True   # ✅ allow meta-learner to see both raw features + base preds
)

# -------------------------------
# 4. Train & Predict
# -------------------------------
stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)
y_proba = stack_model.predict_proba(X_test)[:, 1]

# -------------------------------
# 5. Evaluation Metrics
# -------------------------------
print("🎯 Final Accuracy:", accuracy_score(y_test, y_pred))
print("\n📑 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# 6. Confusion Matrix Heatmap
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Potable", "Potable"],
            yticklabels=["Not Potable", "Potable"])
plt.title("Confusion Matrix - Improved Stacking Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 7. ROC-AUC Curve
# -------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve - Improved Stacking Ensemble")
plt.legend(loc="lower right")
plt.show()
