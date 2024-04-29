# Ensemble Methods in Machine Learning
This repository contains two comprehensive Jupyter notebooks that explore different ensemble techniques in machine learning. These notebooks demonstrate how ensemble methods can be used to improve model predictions by combining multiple models' strengths. The first notebook covers ensemble techniques like hard voting, bagging, and random forests, while the second focuses on boosting methods like AdaBoost and Gradient Boost.

## Overview
The notebooks are designed to provide a thorough educational exploration into ensemble methods, detailing their theoretical backgrounds and practical applications:

- Ensemble Methods: Introduces basic ensemble techniques and applies them to classification tasks using the Census Income and Breast Cancer datasets.

- Ensemble Boost: Focuses on boosting methods, discussing their advantages and applications on the same datasets.
Notebooks

### 1. Ensemble Methods
File: ensemble_methods.ipynb

Description:
This notebook delves into various basic ensemble techniques, including:

- Hard Voting: Combines different classification models based on a majority vote for prediction.

- Bagging: Uses bootstrapped datasets to create multiple models and aggregates their predictions to improve the overall result.

- Random Forests: An extension of bagging applied to decision trees, optimizing both model variance and bias.

Datasets Used:
- Census Income Dataset: Predicts whether income exceeds $50K/yr based on census data.

- Breast Cancer Dataset: Classifies cancer diagnosis as benign or malignant.

### 2. Ensemble Boost
File: ensemble_boost.ipynb

Description:
This notebook focuses on advanced ensemble boosting techniques:

- AdaBoost: Improves classification accuracy by adjusting weights of incorrectly classified instances and combining weak learners.

- Gradient Boost: Uses successive models to correct errors of previous models, with a focus on minimizing a loss function.

Datasets Used:
- Census Income Dataset: Demonstrates how boosting methods can enhance predictions in complex classification scenarios.

- Breast Cancer Dataset: Shows the effectiveness of boosting in medical diagnostic accuracy.