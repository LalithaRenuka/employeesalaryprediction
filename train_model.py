import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Load data
df = pd.read_csv(r"C:\Users\lkoti\OneDrive\Documents\salary_predictor_india\adult 3.csv")

# Select required columns
df = df[['age', 'workclass', 'education', 'marital-status', 'occupation',
         'gender', 'hours-per-week', 'native-country', 'income']]

df.dropna(inplace=True)

# Define features and target
X = df.drop("income", axis=1)
y = df["income"]

# Preprocessing
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'gender', 'native-country']
numerical_cols = ['age', 'hours-per-week']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

# Confusion Matrix (Red theme)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap="Reds", ax=ax)
plt.title("Confusion Matrix", color='darkred')
plt.savefig("confusion_matrix.png")
plt.close()

# Feature importance (Red theme)
importances = model.named_steps['classifier'].feature_importances_
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]

plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp, y=feat_imp.index, palette="Reds_r")
plt.title("Top 10 Feature Importances", color='darkred')
plt.xlabel("Importance", color='darkred')
plt.ylabel("Features", color='darkred')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Save model
joblib.dump(model, "income_classifier_model.joblib")
print("✅ Model saved as income_classifier_model.joblib")
