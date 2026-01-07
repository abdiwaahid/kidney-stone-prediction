# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve)

# XGBoost
from xgboost import XGBClassifier

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully.")


# Cell 2: Load Data and Filter Features

# 1. Load the dataset
# Replace 'Kidney_stone_dataset.csv' with your specific file path if different
df = pd.read_csv('data/cleaned_stone.csv')

# 2. Define the strict feature set (The "Honest" List)
# We exclude 'hematuria', 'ana', 'months' as they are potential leakage/outcomes
honest_features = [
    'blood_pressure', 'water_intake', 'physical_activity', 'diet',
    'smoking', 'alcohol', 'painkiller_usage', 'family_history',
    'weight_changes', 'stress_level'
]

target = 'stone_risk'

# 3. Create Feature Matrix (X) and Target Vector (y)
X = df[honest_features].copy()
y = df[target].copy()

print(f"Original Dataset Shape: {df.shape}")
print(f"Filtered Feature Shape: {X.shape}")
print("-" * 30)
print("Final Features being used:", X.columns.tolist())


# Cell 3: Check Target Distribution and Data Types
print("Target Distribution:")
print(y.value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution of Kidney Stone Risk (Target)")
plt.xlabel("Risk (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Identify Numerical vs Categorical columns automatically
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# Cell 4: Define Preprocessing Pipeline

# Create the ColumnTransformer
# This handles scaling for numbers and encoding for strings simultaneously
preprocessor = ColumnTransformer(
    transformers=[
        # Numerical: Median imputation + Scaling
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        
        # Categorical: Frequent imputation + One-Hot Encoding
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

print("Preprocessing pipeline defined.")


# Cell 5: Stratified Train-Test Split

# We use stratify=y to maintain the class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

# Cell 6: Feature Selection (Permutation Importance) & Data Filtering
from sklearn.inspection import permutation_importance

print("Running Permutation Feature Importance...")

# 1. Transform the training data (Model needs numbers)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test) # Transform test data too

# 2. Train the "Judge" Model (Random Forest)
judge_model = RandomForestClassifier(n_estimators=100, random_state=42)
judge_model.fit(X_train_processed, y_train)

# 3. Calculate Permutation Importance
perm_importance = permutation_importance(
    judge_model, 
    X_train_processed, 
    y_train, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1
)

# 4. Identify Top Features
# Get the names of the columns after OneHotEncoding
try:
    feature_names_out = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names_out = numerical_features + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features))

# Create a dataframe of importance
importance_df = pd.DataFrame({
    'Feature': feature_names_out,
    'Importance': perm_importance.importances_mean
})

# Sort and pick Top 10
top_features_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
top_feature_indices = top_features_df.index.values # The numerical indices of columns to keep

print("Selected Final Features:")
print(top_features_df['Feature'].values)

# 5. CREATE THE FINAL REDUCED DATASETS
# We slice the processed numpy arrays to keep ONLY the top columns
X_train_final = X_train_processed[:, top_feature_indices]
X_test_final = X_test_processed[:, top_feature_indices]

print(f"\nOriginal Feature Count: {X_train_processed.shape[1]}")
print(f"Final Feature Count: {X_train_final.shape[1]}")

# Cell 7: Define Models and Search Space for Optimization

# 1. Define the 3 Algorithms
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# 2. Define Hyperparameter Grids
# NOTE: We removed 'classifier__' prefix because we are fitting models directly, not via Pipeline
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.9, 1.0]
    }
}
print("Models and Grids defined (Prefixes removed).")

# Cell 8: Training and Hyperparameter Tuning on FINAL Features

best_estimators = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Optimization on Selected Features...")

for name, model in models.items():
    print(f"Training {name}...")
    
    # RandomizedSearchCV: Finds best params efficiently
    search = RandomizedSearchCV(
        model,                 # Pass model directly (no Pipeline)
        param_distributions=param_grids[name], 
        n_iter=15,             
        cv=cv,                 
        scoring='roc_auc',     
        n_jobs=-1,             
        random_state=42
    )
    
    # Fit on the FINAL (Reduced) data
    search.fit(X_train_final, y_train)
    
    best_estimators[name] = search.best_estimator_
    print(f"  -> Best AUC: {search.best_score_:.4f}")
    print(f"  -> Best Params: {search.best_params_}\n")

print("Optimization Complete.")


# Cell 9: Detailed Evaluation on Test Set

results_table = []

print("Evaluating on Test Set...")

for name, model in best_estimators.items():
    # Predict classes and probabilities
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Calculate Sensitivity & Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Store results
    results_table.append({
        'Model': name,
        'Accuracy': acc,
        'Balanced Acc': bal_acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'AUC': auc
    })

# Create DataFrame for the report
results_df = pd.DataFrame(results_table)
results_df = results_df.sort_values(by='AUC', ascending=False)

print("\n--- Comparative Analysis Results ---")
display(results_df) # Use display() for pretty printing in Jupyter, or print()


# Cell 10: Plot Combined ROC Curves

plt.figure(figsize=(10, 8))

for name, model in best_estimators.items():
    y_prob = model.predict_proba(X_test_final)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

# Plot Random Chance Line
plt.plot([0, 1], [0, 1], 'k--', linestyle='--', color='gray', label='Random Chance')

# Aesthetics
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Combined ROC Curves for Kidney Stone Risk', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)

plt.show()


# Cell: EDA - Descriptive Stats and Distributions

# 1. Generate Table 1: Descriptive Statistics
# We separate the data by target to see differences
desc_stats = df.groupby(target)[honest_features].describe().transpose()
print("--- Table 1: Descriptive Statistics by Stone Risk ---")
display(desc_stats)

# 2. Visualize Key Numerical Distributions
# We focus on the numerical honest features
num_cols_eda = df[honest_features].select_dtypes(include=['number']).columns

plt.figure(figsize=(15, 5))
for i, col in enumerate(num_cols_eda):
    plt.subplot(1, len(num_cols_eda), i+1)
    sns.boxplot(x=target, y=col, data=df, palette='Set2')
    plt.title(f'{col} vs Risk')
    plt.xlabel('Stone Risk (0=No, 1=Yes)')
plt.tight_layout()
plt.show()

# 3. Visualize Key Categorical Distributions
cat_cols_eda = ['diet', 'physical_activity', 'family_history'] # Select top 3 interesting ones

plt.figure(figsize=(15, 5))
for i, col in enumerate(cat_cols_eda):
    plt.subplot(1, 3, i+1)
    sns.countplot(x=col, hue=target, data=df, palette='pastel')
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cell: Correlation Heatmap of Selected Features

# Reconstruct a DataFrame from the selected final features for visualization
# Note: We use the names of the top features we found in the Feature Selection step
X_train_final_df = pd.DataFrame(X_train_final, columns=top_features_df['Feature'].values)

plt.figure(figsize=(12, 10))
# Calculate correlation matrix
corr_matrix = X_train_final_df.corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Final Selected Features')
plt.show()

# Cell: Feature Importance Plot

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
plt.title('Top 10 Predictors (Permutation Importance)')
plt.xlabel('Drop in Accuracy when Shuffled')
plt.ylabel('Feature Name')
plt.grid(axis='x', alpha=0.3)
plt.show()

# Cell: Confusion Matrix Comparison
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, model) in enumerate(best_estimators.items()):
    # Predict on Final Test Set
    y_pred = model.predict(X_test_final)
    
    # Plot
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, 
        y_pred, 
        ax=axes[i], 
        cmap='Blues', 
        colorbar=False
    )
    axes[i].set_title(f'{name} Confusion Matrix')

plt.tight_layout()
plt.show()


# Cell: SHAP Interpretation
import shap

# 1. Select the best model (assuming XGBoost is usually best, or pick specifically)
best_model_name = 'XGBoost' 
final_model = best_estimators[best_model_name]

# 2. Create SHAP Explainer
# Since we are using XGBoost, we use TreeExplainer
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_final)

# 3. Generate Summary Plot
print(f"Generating SHAP Plot for {best_model_name}...")
plt.figure()
shap.summary_plot(
    shap_values, 
    X_train_final, 
    feature_names=top_features_df['Feature'].values,
    show=True
)

# Cell: SAVE ALL OUTPUTS
import os

# 1. Create a folder to store results
output_folder = 'project_outputs'
os.makedirs(output_folder, exist_ok=True)
print(f"Created folder: {output_folder}")

# ==========================================
# PART A: SAVE TABLES (as CSV files)
# ==========================================

# 1. Descriptive Statistics (Table 1)
desc_stats_export = df.groupby(target)[honest_features].describe().transpose()
desc_stats_export.to_csv(f'{output_folder}/table1_descriptive_stats.csv')
print("Saved: table1_descriptive_stats.csv")

# 2. Model Comparison Results (The Final Scoreboard)
# results_df was created in the evaluation step
results_df.to_csv(f'{output_folder}/table2_model_results.csv', index=False)
print("Saved: table2_model_results.csv")

# 3. Selected Features List
pd.DataFrame(top_features_df['Feature']).to_csv(f'{output_folder}/list_selected_features.csv', index=False)
print("Saved: list_selected_features.csv")


# ==========================================
# PART B: SAVE PLOTS (as High-Res PNGs)
# ==========================================

# NOTE: dpi=300 makes the image high resolution (great for Word/PDF reports)
# bbox_inches='tight' ensures no labels get cut off

# --- Figure 1: Correlation Matrix ---
plt.figure(figsize=(12, 10))
# Reconstruct DF for plotting
X_train_final_df = pd.DataFrame(X_train_final, columns=top_features_df['Feature'].values)
sns.heatmap(X_train_final_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.savefig(f'{output_folder}/fig1_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close() # Close plot to free memory
print("Saved: fig1_correlation_matrix.png")

# --- Figure 2: Feature Importance Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=top_features_df, palette='viridis')
plt.title('Top 10 Predictors (Permutation Importance)')
plt.xlabel('Drop in Accuracy')
plt.grid(axis='x', alpha=0.3)
plt.savefig(f'{output_folder}/fig2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_feature_importance.png")

# --- Figure 3: Combined ROC Curves ---
plt.figure(figsize=(10, 8))
for name, model in best_estimators.items():
    y_prob = model.predict_proba(X_test_final)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_folder}/fig3_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3_roc_curves.png")

# --- Figure 4: Confusion Matrices ---
from sklearn.metrics import ConfusionMatrixDisplay
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, model) in enumerate(best_estimators.items()):
    y_pred = model.predict(X_test_final)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{name}')
plt.tight_layout()
plt.savefig(f'{output_folder}/fig4_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig4_confusion_matrices.png")

# --- Figure 5: SHAP Summary Plot ---
# SHAP is special, we need to create a new figure context for it
import shap
best_model_name = 'XGBoost' # Ensure this matches your best model key
if best_model_name in best_estimators:
    final_model = best_estimators[best_model_name]
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train_final)
    
    # shap.summary_plot doesn't return a figure, it plots to the current figure
    # We pass show=False so we can save it
    plt.figure()
    shap.summary_plot(shap_values, X_train_final, feature_names=top_features_df['Feature'].values, show=False)
    plt.savefig(f'{output_folder}/fig5_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_shap_summary.png")

print(f"\nDONE! Check the folder '{output_folder}' for all your files.")

# Cell: SAVE INDIVIDUAL EDA CHARTS
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create specific folder for EDA images to keep things organized
eda_folder = 'project_outputs/eda_charts'
os.makedirs(eda_folder, exist_ok=True)
print(f"Saving charts to folder: {eda_folder}/")

# 2. Define Feature Groups
# Numerical features (for Boxplots)
num_cols_eda = df[honest_features].select_dtypes(include=['number']).columns

# Categorical features (for Countplots)
# We treat any non-numerical column as categorical
cat_cols_eda = [col for col in honest_features if col not in num_cols_eda]


# ==========================================
# PART A: SAVE NUMERICAL CHARTS (Boxplots)
# ==========================================
print("\n--- Saving Numerical Charts ---")
for col in num_cols_eda:
    plt.figure(figsize=(6, 5))
    
    # Plot Boxplot: Comparison of distribution between Risk (0) and Risk (1)
    sns.boxplot(x=target, y=col, data=df, palette='Set2')
    
    plt.title(f'{col} vs Kidney Stone Risk')
    plt.xlabel('Stone Risk (0=No, 1=Yes)')
    plt.ylabel(col)
    
    # Save file
    filename = f"{eda_folder}/num_{col}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close to free memory
    print(f"Saved: {filename}")


# ==========================================
# PART B: SAVE CATEGORICAL CHARTS (Countplots)
# ==========================================
print("\n--- Saving Categorical Charts ---")
for col in cat_cols_eda:
    plt.figure(figsize=(8, 6))
    
    # Plot Countplot: Frequency of categories split by Risk
    sns.countplot(x=col, hue=target, data=df, palette='pastel')
    
    plt.title(f'{col} Distribution by Risk')
    plt.xlabel(col)
    plt.ylabel('Count of Patients')
    plt.xticks(rotation=45) # Rotate labels if they are long
    plt.legend(title='Risk (0/1)', loc='upper right')
    
    # Save file
    filename = f"{eda_folder}/cat_{col}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

print(f"\nAll individual charts saved in: {eda_folder}")