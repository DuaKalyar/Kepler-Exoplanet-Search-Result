#  PROJECT TOPIC:END-TO-END EXOPLANET CLASSIFICATION
# ============================================================

# --- Step 0: Installing the shap and lime and xgboost libraries
!pip install shap lime xgboost --quiet

# --- Step 1: Importing all essentail libraries which are used in my project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from google.colab import files
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  Step 2: We will upload our data set
# ============================================================
print(" Please upload your dataset (e.g., cumulative.csv)")
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
print(" Dataset Loaded! Shape:", df.shape)
df.head()

# ============================================================
#  Step 3: FULL EDA (Exploratory Data Analysis)
# ============================================================
print("\n Starting Exploratory Data Analysis...\n")

# --- 1‼ Dataset Info ---
print("Dataset Info:\n")
print(df.info())

# --- 2‼ Missing Value Heatmap showing missing values---
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.show()

# --- 3‼ Target / Label Distribution of target label koi disposition ---
if 'koi_disposition' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='koi_disposition', palette='Set2')
    plt.title("Label Distribution")
    plt.show()

# --- 4‼ Numeric Feature Distribution ---
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 0:
    numeric_df.hist(
        bins=30,
        figsize=(18,16),
        layout=(int(np.ceil(len(numeric_df.columns)/4)), 4),
        edgecolor='black'
    )
    plt.suptitle("Numeric Feature Distributions", fontsize=18, y=0.93)
    plt.tight_layout(pad=3.0)
    plt.show()

# --- 5‼ Correlation Heatmap (numeric only) ---
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap (Numeric Only)")
    plt.show()

print(" EDA completed!\n")

# ============================================================
#  Step 4: Data Preprocessing and cleaning dataset
# ============================================================
# Drop fully empty columns
df = df.dropna(axis=1, how='all')

# Fill missing values
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].mean(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Label column
label_col = 'koi_disposition'  # change if needed
X = df.drop(columns=[label_col])
y = df[label_col]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance data with SMOTE(synthetic minority over sampling technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("After SMOTE:", X_res.shape, y_res.shape)

# ============================================================
#  Step 5: Feature Selection of dataset INfo gain entropy and giniindex
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print("\n Performing Feature Selection...")

# --- Information Gain (Based on entropy how much uncertanity is reduced)---
info_gain = mutual_info_classif(X_train, y_train)
info_df = pd.DataFrame({'Feature': X.columns, 'Info_Gain': info_gain}).sort_values(by='Info_Gain', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Info_Gain', y='Feature', data=info_df.head(15), palette="Blues_d")
plt.title("Top 15 Features by Information Gain")
plt.show()

# --- Gini Index (measure of impurity)---
dt_gini = DecisionTreeClassifier(random_state=42)
dt_gini.fit(X_train, y_train)
gini_df = pd.DataFrame({'Feature': X.columns, 'Gini_Importance': dt_gini.feature_importances_}).sort_values(by='Gini_Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Gini_Importance', y='Feature', data=gini_df.head(15), palette="Greens_d")
plt.title("Top 15 Features by Gini Importance")
plt.show()

# --- Entropy(measure of uncertanity) ---
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)
entropy_df = pd.DataFrame({'Feature': X.columns, 'Entropy_Importance': dt_entropy.feature_importances_}).sort_values(by='Entropy_Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Entropy_Importance', y='Feature', data=entropy_df.head(15), palette="Oranges_d")
plt.title("Top 15 Features by Entropy Importance")
plt.show()

# ============================================================
#  Step 6: Model Training + Feature Ranking
# ============================================================
print("\n Training Models and Plotting Feature Rankings...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "KNN": KNeighborsClassifier()  # Added KNN for completeness
}

# Train all models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

print("\nAll models trained successfully!\n")

# ============================================================
#  PERFORMANCE EVALUATION (FULL METRICS)
# ============================================================

performance_rows = []
roc_curves = {} # Store ROC data

print("\n Detailed Performance Metrics:\n")

classes = np.unique(y_test)
n_classes = len(classes)

for name, model in models.items():
    preds = model.predict(X_test)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T  # Fix binary probs

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) != 0 else 0

    auc = None
    if probs is not None:
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')

    cm = confusion_matrix(y_test, preds)
    print(f"\n Confusion Matrix for {name}:\n{cm}")
    performance_rows.append([name, acc, prec, rec, f1, f2, auc, cm])

    # ROC curves
    if probs is not None:
        model_roc_curves_data = []
        for i, class_id in enumerate(classes):
            y_test_bin = (y_test == class_id).astype(int)
            score = probs[:, i] if probs.shape[1] > 1 else probs[:, 0]
            fpr, tpr, _ = roc_curve(y_test_bin, score)
            model_roc_curves_data.append((fpr, tpr, f"Class {class_id}"))
        roc_curves[name] = model_roc_curves_data

performance_df = pd.DataFrame(
    performance_rows,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score", "AUC", "Confusion Matrix"]
)

print("\n\n Combined Performance Table:")
print(performance_df)

# ============================================================
#  ROC CURVES FOR ALL MODELS
# ============================================================

plt.figure(figsize=(12,8))
for name, model_curves in roc_curves.items():
    for fpr, tpr, class_label in model_curves:
        plt.plot(fpr, tpr, label=f"{name} ({class_label})")

plt.plot([0,1],[0,1],'--', color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (All Models - One-vs-Rest)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
#  FEATURE IMPORTANCE VISUALS FOR EACH MODEL
# ============================================================

feature_names = X.columns.tolist()

for name, model in models.items():
    print(f"\n🔍 Feature Importance for {name}")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            plt.figure(figsize=(10, 5))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.title(f"Feature Importance - {name}")
            plt.tight_layout()
            plt.show()
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        if len(importances) == len(feature_names):
            plt.figure(figsize=(10, 5))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.title(f"Feature Importance (Coefficients) - {name}")
            plt.tight_layout()
            plt.show()
    else:
        print(f" {name} does not support direct feature importance.")

# ============================================================
#  ACCURACY COMPARISON VISUAL(Graph showing accuracy)
# ============================================================

plt.figure(figsize=(10,6))
plt.bar(performance_df["Model"], performance_df["Accuracy"])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# ============================================================
#  BEST MODEL SELECTION
# ============================================================

best_row = performance_df.loc[performance_df["Accuracy"].idxmax()]
best_model_name = best_row["Model"]
best_accuracy = best_row["Accuracy"]
best_model = models[best_model_name]

print(f"\n BEST MODEL: {best_model_name} with Accuracy = {best_accuracy:.4f}")

# ============================================================
#  Step 7: Interpretability (SHAP + LIME)
# ============================================================

print(f"\n Interpreting {best_model_name} with SHAP & LIME...")

X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# --- SHAP ---
explainer = shap.Explainer(best_model, X_train_df)
shap_values = explainer(X_test_df[:100])
shap.summary_plot(shap_values, features=X_test_df[:100], feature_names=X.columns)

# --- LIME ---
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in classes],
    mode='classification'
)

i = np.random.randint(0, X_test.shape[0])
exp = explainer_lime.explain_instance(X_test[i], best_model.predict_proba, num_features=10, top_labels=1)
exp.show_in_notebook(show_table=True)

print("\n End-to-End Exoplanet Mining Pipeline Completed Successfully!")
