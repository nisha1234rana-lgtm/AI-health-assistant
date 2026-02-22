import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---- SETTINGS ----
DATA_PATH = "synthetic_symptom_disease_30_v2.csv"  # <-- put your dataset here
TARGET_COL = "prognosis"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---- LOAD DATA ----
df = pd.read_csv(DATA_PATH)

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- TRAIN MODEL ----
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

# ---- EVALUATE ----
pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Random Forest Test Accuracy:", acc)

# ---- SAVE ARTIFACTS ----
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_disease_model.joblib"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "symptom_columns.joblib"))

print("Saved model to models/rf_disease_model.joblib")
print("Saved symptom columns to models/symptom_columns.joblib")