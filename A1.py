# https://github.com/Agekyan/CS4680/blob/main/A1.md # ... Findings at the bottom of this .py
# !!! #
# Assignment 1
# David Agekyan
# Professor Sun


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('/content/heart.csv')
print("Dataset loaded.")

# 1. Check for missing values - Already done in df.info(), no missing values found.
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Identify and treat outliers - cap 99th percentile for upper bounds and 1st percentile for lower bounds.
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numerical_cols:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
print("\nOutliers treated using capping.")

# 3. Encode categorical variables
# The columns sex, cp, fbs, restecg, exang, slope, ca, and thal are treated as categorical.
# Use one-hot encoding <<!!!
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nCategorical features encoded.")

# 4. Scale numerical features .. identify numerical columns again after potential outlier treatment
numerical_cols_after_encoding = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
df_processed[numerical_cols_after_encoding] = scaler.fit_transform(df_processed[numerical_cols_after_encoding])
print("\nNumerical features scaled.")

# 5. Separate features (X) and target variable (y)
X = df_processed.drop('target', axis=1)
y = df_processed['target']
print("\nFeatures (X) and target (y) separated.")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # Using stratify for balanced target distribution
print("\nData split into training and testing sets (75% train, 25% test).")
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Model Development KNN, SVM, Decision Trees, Random Forest .. Instantiate the chosen regression models
knn_model = KNeighborsClassifier()
svm_model = SVC()
dt_model = DecisionTreeClassifier(random_state=42) # random_state for reproducibility
rf_model = RandomForestClassifier(random_state=42) # .

# Train on training data
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
print("\nAll models trained successfully:")
print("- KNN Model")
print("- SVM Model")
print("- Decision Tree Model")
print("- Random Forest Model")


# Model evaluation
# Evaluate the performance focusing on recall for the positive class.

# Make predictions on the testing data for each model
y_pred_knn = knn_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluate and print metrics for...

# KNN
print("\n--- KNN Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_knn):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_knn):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("-" * 28)

# SVM
print("\n--- SVM Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_svm):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("-" * 28)

# Decision Tree
print("\n--- Decision Tree Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_dt):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("-" * 28)

# Random Forest
print("\n--- Random Forest Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("-" * 28)


# Compare the performance of the different models and discuss their suitability for the problem.
model_performance = {
    'KNN': {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'Precision': precision_score(y_test, y_pred_knn),
        'Recall': recall_score(y_test, y_pred_knn),
        'F1-score': f1_score(y_test, y_pred_knn)
    },
    'SVM': {
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm),
        'Recall': recall_score(y_test, y_pred_svm),
        'F1-score': f1_score(y_test, y_pred_svm)
    },
    'Decision Tree': {
        'Accuracy': accuracy_score(y_test, y_pred_dt),
        'Precision': precision_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt),
        'F1-score': f1_score(y_test, y_pred_dt)
    },
    'Random Forest': {
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-score': f1_score(y_test, y_pred_rf)
    }
}
print("\n## Model Performance Summary")
for model_name, metrics in model_performance.items():
    print(f"\n--- {model_name} ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Findings, Eval, Metrics, Misc

# When predicting heart disease, Recall is a crucial metric.
# A high Recall score indicates that the model is good at identifying most of the actual positive
# cases (people with heart disease), minimizing false negatives (missing a diagnosis). Missing a
# heart disease diagnosis can have severe consequences, making a higher Recall generally preferable
# even if it comes at the cost of slightly lower Precision (more false positives).

# The model with the highest Recall is the best performing for capturing true positive cases of heart disease.
# Decision Tree, Random Forest - relatively strong Recall scores.
# SVM model lower, misses more actual positive cases. KNN model’s performance varies, variable recall

# Models with higher Recall (like Random Forest) may have lower Precision, meaning they might
# incorrectly diagnose some healthy individuals with heart disease (false positives). While
# undesirable, false positives are generally less critical than false negatives in this context,
# as they would lead to further testing rather than a missed diagnosis.

# ! Models with higher Precision might have lower Recall, missing actual cases of heart disease !

# Both the Decision Tree and Random Forest models achieved the highest recall (0.8537),  
# making them most effective for identifying true heart disease cases and minimizing missed diagnoses—  
# a key priority in this medical context.  
# The SVM model showed moderately high recall (0.8293) and the best F1-score (0.8000),  
# suggesting it maintains the most balanced trade-off between precision and recall,  
# which could be useful if reducing false positives is also important.  
# KNN had the lowest recall (0.7805) and F1-score (0.7711),  
# indicating it is less suitable for this problem where minimizing false negatives is critical.
