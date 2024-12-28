import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('mushrooms.csv')

# Split target and features
X = df.drop('class', axis=1)
y = df['class']

# Apply OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = onehot.fit_transform(X)

# Get feature names for visualization
feature_names = onehot.get_feature_names_out(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("\nCross-validation Scores:")
print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train the model
model.fit(X_train, y_train)

# Compare training and testing scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTraining score: {train_score:.4f}")
print(f"Testing score: {test_score:.4f}")

# Check for overfitting
if train_score - test_score > 0.1:
    print("\nWarning: Model might be overfitting (large gap between training and test scores)")

# Feature importance analysis with better visualization
plt.style.use('default')  # Changed from 'seaborn' to 'default'

# Sort features by importance and get top 15 features
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
})
top_features = feature_importance.sort_values('importance', ascending=True).tail(15)

# Create figure with higher resolution and better size
plt.figure(figsize=(12, 8), dpi=100)

# Create horizontal bar plot with better colors
bars = plt.barh(y=range(len(top_features)), width=top_features['importance'], 
        color='#2ecc71', edgecolor='#27ae60')

# Customize the plot
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold', pad=20)

# Add value labels on the bars
for i, v in enumerate(top_features['importance']):
    plt.text(v, i, f' {v:.3f}', va='center', fontsize=10)

# Add grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()