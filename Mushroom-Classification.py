import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('mushrooms.csv')

# Split target and features
X = df.drop('class', axis=1)
y = df['class']

# Apply OneHotEncoder
onehot = OneHotEncoder(sparse_output=False,)
X_encoded = onehot.fit_transform(X)

# Get feature names for visualization
feature_names = onehot.get_feature_names_out(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Farklı max_depth değerleri için skorları hesaplayalım
max_depths = range(1, 5)
train_scores = []
test_scores = []

print("\nAccuracy scores for different max_depths:")
print("----------------------------------------")
for depth in max_depths:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=depth)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f"Max Depth {depth:2d} -> Train: {train_score:.4f}, Test: {test_score:.4f}")

# Plot training vs testing scores
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, 'o-', color='#2ecc71', label='Training Score', linewidth=2)
plt.plot(max_depths, test_scores, 'o-', color='#e74c3c', label='Testing Score', linewidth=2)
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Training vs Testing Scores for Different Tree Depths', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# Cross-validation scores için görselleştirme
cv_scores = []
models = []

print("\nCross-validation scores for different max_depths:")
print("-----------------------------------------------")
for depth in max_depths:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=depth)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_scores.append(scores)
    print(f"Max Depth {depth:2d} -> CV mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    model.fit(X_train, y_train)
    models.append(model)

# Box plot - güncellenmiş parametre adıyla
plt.figure(figsize=(12, 6))
plt.boxplot(cv_scores, tick_labels=max_depths)  # 'labels' yerine 'tick_labels'
plt.xlabel('Max Depth', fontsize=12)
plt.ylabel('Cross-validation Score', fontsize=12)
plt.title('Cross-validation Scores Distribution for Different Tree Depths', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# En iyi modeli seçelim (test skoru en yüksek olan)
best_idx = np.argmax(test_scores)
best_model = models[best_idx]

# Best model'in performans metrikleri
print("\nBest Model Performance (max_depth =", max_depths[best_idx], "):")
print("------------------------------------------------")
print(f"Training accuracy: {best_model.score(X_train, y_train):.4f}")
print(f"Testing accuracy: {best_model.score(X_test, y_test):.4f}")

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("\nCross-validation Scores:")
print(f"CV scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Check for overfitting
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
if train_score - test_score > 0.1:
    print("\nWarning: Model might be overfitting (large gap between training and test scores)")

# Feature importance kısmını düzeltelim - tekrar eden kısmı silelim
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
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

# Confusion Matrix - son plot olarak
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['edible', 'poisonous'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Best Model', fontsize=14, pad=15)
plt.tight_layout()
plt.show()
