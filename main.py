import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import shap

df = pd.read_csv('data/cleaned_stone.csv')

leaky_columns = ['cluster', 'ckd_pred', 'ckd_stage', 'months', 'hematuria', 'ana']
honest_features = [
    'blood_pressure', 'water_intake', 'physical_activity', 'diet',
    'smoking', 'alcohol', 'painkiller_usage', 'family_history',
    'weight_changes', 'stress_level'
]
target = 'stone_risk'

df = df.drop(columns=leaky_columns)

print(df.head())
print(df.info())
print(df.describe())

# ======== Data Analysis ========
print(df['stone_risk'].value_counts(normalize=True))
sns.countplot(x='stone_risk', data=df)
plt.title('Distribution of Kidney Stone Risk')
plt.show()

# ======== Seperate Features and Target ========
X = df[honest_features]
y = df[target]

# ======== Identify Numerical and Categorical Features ========
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)


# ======== Split Data ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======== Preproccesing piplines ========
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# ======== Model Selection ========

# Logistic Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# train models
lr_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

print("Logistic Regression Score:", lr_pipeline.score(X_test, y_test))
print("XGBoost Score:", xgb_pipeline.score(X_test, y_test))

# ======== Model Evaluation ========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(xgb_pipeline, X, y, cv=cv, scoring='roc_auc')

print(f"XGBoost Cross-Validation ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")


# ======== Perform Feature Selection ========
rfe_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rfe_model)])

# top 10 features
rfe = RFE(estimator=rfe_model, n_features_to_select=10, step=1)
rfe_selection_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('rfe', rfe)])
rfe_selection_pipeline.fit(X_train, y_train)
selected_feature_names = rfe_selection_pipeline.named_steps['preprocessor'].get_feature_names_out()[rfe_selection_pipeline.named_steps['rfe'].support_]
print("Top 10 selected features:", selected_feature_names)

# ======== Final Evaluation ========
param_dist = {
    'classifier__n_estimators': [100, 200, 300, 400],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=param_dist, n_iter=25,
    scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)
print("Best cross-validation ROC-AUC:", random_search.best_score_)
best_model = random_search.best_estimator_


# ======== Evaluate Final Model on Test Set ========
y_pred = best_model.predict(X_test)

print("\n--- Final Model Evaluation on Test Set ---")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title('Final Model Confusion Matrix')
plt.show()


# ======== Interpretation and Reporting ========

# SHAP 
X_train_processed = best_model.named_steps['preprocessor'].transform(X_train)
explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
shap_values = explainer.shap_values(X_train_processed)

# feature names 
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()

# summary plot
shap.summary_plot(shap_values, X_train_processed, feature_names=feature_names)
plt.title('Feature Importance')
plt.show()