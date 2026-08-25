import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# --- 1. LOAD DATASET ---
df = pd.read_csv('bank-full.csv', sep=';')  # Adjust separator if needed (e.g., ',' or ';')

# Target variable encoding: 'yes' -> 1, 'no' -> 0
df['y'] = df['y'].map({'yes': 1, 'no': 0})

X = df.drop('y', axis=1)
y = df['y']

# Define categorical and numerical feature sets
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- 2. TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. PREPROCESSING PIPELINE ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline combining preprocessor and Logistic Regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# --- 4. MODEL TRAINING ---
model_pipeline.fit(X_train, y_train)

# --- 5. EVALUATION ---
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Subscription', 'Subscribed']))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Feature Importance extraction
onehot_cols = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = numerical_cols + list(onehot_cols)
coefficients = model_pipeline.named_steps['classifier'].coef_[0]

coef_df = pd.DataFrame({'Feature': all_features, 'Coefficient': coefficients})
coef_df['Absolute_Coef'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Absolute_Coef', ascending=False)

print("\n--- Top 10 Influential Features ---")
print(coef_df[['Feature', 'Coefficient']].head(10))