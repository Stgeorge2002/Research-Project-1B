# Install the required libraries (run this command in your terminal or command prompt)
# pip install numpy pandas scikit-learn xgboost scipy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr

# Specify the path to your dataset file (replace with your actual file path)
dataset_path = "path/to/your/dataset.csv"

# Read the dataset from the specified file path
data = pd.read_csv(dataset_path)

# Separate the features (X) and the labeling column (y)
X = data.drop("labeling_column", axis=1)  # Replace "labeling_column" with the actual column name
y = data["labeling_column"]  # Replace "labeling_column" with the actual column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Use XGBoost's built-in feature importance for initial feature selection
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
selector = SelectFromModel(xgb, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Step 2: Apply Recursive Feature Elimination (RFE) with XGBoost
rfe = RFE(estimator=XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42), n_features_to_select=int(X_train_selected.shape[1] * 0.8), step=1)
rfe.fit(X_train_selected, y_train)
X_train_rfe = rfe.transform(X_train_selected)
X_test_rfe = rfe.transform(X_test_selected)

# Step 3: Use SelectKBest with f_classif for univariate feature selection
selector = SelectKBest(score_func=f_classif, k=int(X_train_rfe.shape[1] * 0.6))
selector.fit(X_train_rfe, y_train)
X_train_kbest = selector.transform(X_train_rfe)
X_test_kbest = selector.transform(X_test_rfe)

# Step 4: Apply PCA for dimensionality reduction
pca = PCA(n_components=int(X_train_kbest.shape[1] * 0.5))
X_train_pca = pca.fit_transform(X_train_kbest)
X_test_pca = pca.transform(X_test_kbest)

# Step 5: Use Random Forest for feature importance ranking
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
X_train_rf = X_train_pca[:, indices[:int(X_train_pca.shape[1] * 0.8)]]
X_test_rf = X_test_pca[:, indices[:int(X_train_pca.shape[1] * 0.8)]]

# Step 6: Calculate Pearson correlation coefficients and remove highly correlated features
corr_matrix = np.abs(np.array([pearsonr(X_train_rf[:, i], X_train_rf[:, j])[0] for i in range(X_train_rf.shape[1]) for j in range(X_train_rf.shape[1])]).reshape(X_train_rf.shape[1], X_train_rf.shape[1]))
upper_tri = np.triu(corr_matrix, k=1)
to_drop = [column for column in range(upper_tri.shape[1]) if any(upper_tri[:, column] > 0.95)]
X_train_final = np.delete(X_train_rf, to_drop, axis=1)
X_test_final = np.delete(X_test_rf, to_drop, axis=1)

# Train the final XGBoost model with the selected features
xgb_final = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_final.fit(X_train_final, y_train)

# Evaluate the model on the test set
y_pred = xgb_final.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final accuracy: {accuracy:.4f}")
print(f"Number of selected features: {X_train_final.shape[1]}")
