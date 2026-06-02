import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load dataset
df = pd.read_csv('dataset.csv')

# Display dataset info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nChurn distribution:")
print(df['churned'].value_counts())

# Prepare features and target
# Drop customer_id as it's not a feature, keep financial_autonomy_idx as continuous float
X = df[['age', 'financial_autonomy_idx', 'avg_balance']].astype(float)
y = df['churned']

# Verify financial_autonomy_idx is treated as continuous
print("\nFeature data types:")
print(X.dtypes)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Train XGBClassifier with regularization to prevent overfitting
# Regularization constraints: max_depth=3, learning_rate=0.05, subsample=0.8
print("\nTraining XGBClassifier...")
model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Churn', 'Churn']))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': ['age', 'financial_autonomy_idx', 'avg_balance'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# Training accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"\nTrain Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Churn Prediction Model')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("\nROC curve saved as 'roc_curve.png'")

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Churn Prediction Model')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'feature_importance.png'")
