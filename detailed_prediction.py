import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib  # Import joblib to save the model

# Load the dataset
file_path = 'hydrogen_leakage_data.csv'
hydrogen_data = pd.read_csv(file_path)

# Simplify the dataset: Select relevant features and binary target
simplified_data = hydrogen_data[['Hydrogen Level (ppm)', 'Rate of Change (ppm/s)', 'Leakage Status']]

# Map 'Leakage Status' to binary: 1 for 'Leakage' and 0 for other statuses
leakage_mapping = {'Leakage': 1, 'Leakage Stabilized': 0, 'Normal': 0, 'Potential Leakage': 1}
simplified_data['Leakage Status'] = simplified_data['Leakage Status'].map(leakage_mapping)

# Separate features (X) and target (y)
X_simplified = simplified_data[['Hydrogen Level (ppm)', 'Rate of Change (ppm/s)']]
y_simplified = simplified_data['Leakage Status']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_simplified, y_simplified)

# Optional: Standardize the features (scaling can sometimes improve model performance)
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Split the resampled data into training and testing sets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train a Random Forest Classifier on the resampled data
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV (with a broader set of hyperparameters)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Run GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Cross-validation to evaluate the model
cv_scores = cross_val_score(best_rf_model, X_resampled, y_resampled, cv=5)
print(f"Cross-validated accuracy: {cv_scores.mean() * 100:.2f}%")

# Evaluate the model on the test set
y_pred_resampled = best_rf_model.predict(X_test_resampled)
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
classification_report_resampled = classification_report(
    y_test_resampled, y_pred_resampled, target_names=['No Leakage', 'Leakage']
)

# Output the results
print(f"Model Accuracy on Test Data: {accuracy_resampled * 100:.2f}%\n")
print("Classification Report:\n")
print(classification_report_resampled)

# Confusion Matrix for more insights
conf_matrix = confusion_matrix(y_test_resampled, y_pred_resampled)
print(f"Confusion Matrix:\n{conf_matrix}\n")

# ROC-AUC score to evaluate model's ability to distinguish between classes
roc_auc = roc_auc_score(y_test_resampled, best_rf_model.predict_proba(X_test_resampled)[:, 1])
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Feature Importance from the Random Forest Model
feature_importance = best_rf_model.feature_importances_
print(f"Feature Importances: {dict(zip(X_simplified.columns, feature_importance))}")

# Save the trained model to a .pkl file using joblib
joblib.dump(best_rf_model, 'hydrogen_leakage_model.pkl')
print("Model saved to 'hydrogen_leakage_model.pkl'")

