# Mushroom Classification Project

## Overview
The goal of this project is to build a machine learning model to classify mushrooms as either edible or poisonous based on their characteristics. The dataset used for this analysis contains categorical features that describe various properties of mushrooms.

### Dataset
The dataset, `mushrooms.csv`, contains information on:
- **Target Variable:** `class` (edible = 'e', poisonous = 'p')
- **Features:** Various categorical attributes of mushrooms, such as cap shape, color, odor, etc.

## Data Preprocessing
1. **Loading the Data:**
   The dataset was loaded using `pandas`.

2. **Feature Encoding:**
   - Used `OneHotEncoder` to convert categorical features into numerical format.
   - This approach ensures the machine learning model can interpret categorical data effectively.

3. **Splitting the Data:**
   - The data was split into training (70%) and testing (30%) sets using `train_test_split`.

## Model Training
The chosen model for classification is a **Random Forest Classifier**, which is robust and effective for tabular data.

### Steps:
1. **Cross-Validation:**
   - Performed 5-fold cross-validation to evaluate the model's performance on the training set.
   - Results:
     CV scores: [0.99648506 0.98592788 0.98592788 0.99032542 0.99032542]
     Mean CV score: 0.9898 (+/- 0.0078)
2. **Model Fitting:**
   - The model was trained on the training set.
   - Hyperparameters used:
     - `n_estimators`: 100
     - `max_depth`: 3
     - `random_state`: 42

3. **Performance Metrics:**
   - Training score: 0.9905
   - Testing score: 0.9893
   - No significant overfitting observed.

## Feature Importance
A detailed analysis of feature importance was performed to identify the most critical attributes influencing the classification.
- Top 15 features were visualized using a horizontal bar chart.
- Example of important features: [odor, gill-size, gill-color].

## Results
- **Key Findings:**
  - The model performs well on the test set, indicating it generalizes effectively.
  - The top features (e.g., odor, spore print color) are highly indicative of whether a mushroom is edible or poisonous.

- **Potential Model Improvements:**
  - Experiment with other classifiers (e.g., Gradient Boosting, SVM).
  - Fine-tune hyperparameters using grid search or random search.
  - Evaluate the impact of dropping less important features.

## Conclusion
This project successfully demonstrates the use of a Random Forest Classifier for mushroom classification. The workflow includes data preprocessing, model training, evaluation, and feature importance analysis. Future work can involve improving model performance and deploying it in a user-friendly application.

## Files in Repository
1. **Mushroom-Classification.py:** Python script for data processing and model training.
2. **mushrooms.csv:** Dataset used for training and testing the model.

## Additional Notes
A README file has been added to summarize the project structure and its purpose. For more detailed explanations, refer to the Python script and visualization outputs.

