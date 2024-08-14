# Inherently Interpretable Machine Learning for Credit Scoring: Optimal Classification Tree with Hyperplane Splits

Code for the paper *"Inherently Interpretable Machine Learning for Credit Scoring: Optimal Classification Tree with Hyperplane Splits"*.

The code will run in python3 and require [Gurobi11.0](https://www.gurobi.com/products/gurobi-optimizer/) solver.

## Parameters for OCT-H

- **`max_depth`**: This parameter sets the maximum depth of the tree. A deeper tree can capture more complex patterns but may also lead to overfitting and more computation time. The default value is 2.

- **`alpha`**: This parameter controls the complexity of the tree. A larger value of `alpha` results in a sparser tree, which can help prevent overfitting. The default value is 0.01.

- **`N`**: This is the maximum number of features that can be used at a branch node. It also influences the complexity of the tree. The default value is 5.

- **`min_samples_split`**: This parameter specifies the minimum number of instances required at a branch node to consider splitting. The default value is 20.

- **`objective`**: This defines the objective of the OCT-H model.
  - `objective = accuracy`: The model aims to maximize accuracy (OCT-H-Acc).
  - `objective = F1-score`: The model aims to maximize the F1-score (OCT-H-F1).
  - `objective = cost-sensitive`: This is for the cost-sensitive OCT-H (CSOCT-H).
  
  For the credit scoring problem, the default value is `objective = F1-score`.
  

- **`CFN`**: This parameter only functions when `objective = cost-sensitive` and is used to overcome class imbalance. The default value is the ratio of goods to bads.

- **`number_important_features`**: This parameter indicates the number of features selected by the Random Forest (RF) at each branch node. The default value is 15.



## How to Run the Code

After you install the required packages, you can call the main function within a python file as follows (This is what we do in run_exp.py):


import LBC

import OCTH_warm_start

model = OCTH_warm_start.OCTH_warm_start(max_depth=2, alpha=0.01, N=5, objective='F1-score', warmstart=True, output=True)

model.fit(x_train1, y_train)




# Logistic Regression for Credit Scoring

This project implements a Logistic Regression model using the scikit-learn library to evaluate credit scoring. The model is optimized using grid search for hyperparameter tuning.

## Prerequisites

The following Python packages are required to run the code:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`



## Logistic Regression Model Parameters

The Logistic Regression model in this project is implemented using the `LogisticRegression` class from the scikit-learn library. Below is a detailed explanation of the parameters used:

- **`C`**: 
  - **Description**: Inverse of regularization strength; must be a positive float. Regularization is a technique used to prevent overfitting by adding a penalty to the loss function.
  - **Effect**: Smaller values of `C` imply stronger regularization. A larger `C` value means less regularization.
  - **Values Used**: `[0.1, 1, 10]` in grid search.

- **`penalty`**: 
  - **Description**: Specifies the norm used in the penalization.
  - **Options**: 
    - `'l1'`: L1 regularization, which can lead to sparse solutions (some coefficients are exactly zero), useful for feature selection.
    - `'l2'`: L2 regularization, which tends to distribute error across all terms.
  - **Usage**: Both `'l1'` and `'l2'` are explored in grid search.

- **`class_weight`**: 
  - **Description**: Weights associated with classes to handle class imbalance.
  - **Options**: 
    - `None`: All classes are supposed to have weight one.
    - `'balanced'`: Adjusts weights inversely proportional to class frequencies in the input data.
  - **Usage**: Both `None` and `'balanced'` are considered
 

  # Decision Tree Model Parameters

The Decision Tree model in this project is implemented using the `DecisionTreeClassifier` class from the scikit-learn library. Below is a detailed explanation of the parameters used:

- **`criterion`**: 
  - **Description**: The function to measure the quality of a split.
  - **Options**: 
    - `'gini'`: Measures the Gini impurity.
    - `'entropy'`: Measures the information gain.
  - **Usage**: Both `'gini'` and `'entropy'` are explored in grid search.

- **`max_depth`**: 
  - **Description**: The maximum depth of the tree.
  - **Effect**: Controls the maximum number of levels in the tree to prevent overfitting.
  - **Values Used**: `[1, 2, 3, 4, 5, 10]` in grid search.

- **`min_samples_split`**: 
  - **Description**: The minimum number of samples required to split an internal node.
  - **Effect**: Higher values prevent the model from learning overly specific patterns.
  - **Values Used**: `[1, 2, 5, 10, 15, 100]` in grid search.

- **`min_samples_leaf`**: 
  - **Description**: The minimum number of samples required to be at a leaf node.
  - **Effect**: Controls the size of the tree; larger values simplify the model.
  - **Values Used**: `[10, 20, 50, 100]` in grid search.

- **`class_weight`**: 
  - **Description**: Weights associated with classes to handle class imbalance.
  - **Options**: 
    - `None`: All classes have weight one.
    - `'balanced'`: Adjusts weights inversely proportional to class frequencies.
  - **Usage**: Both `None` and `'balanced'` are considered.

The model is optimized using grid search with cross-validation to find the best combination of these parameters based on the ROC AUC score.


# Random Forest Model Parameters

The Random Forest model in this project is implemented using the `RandomForestClassifier` class from the scikit-learn library. Below is a detailed explanation of the parameters used:

- **`n_estimators`**: 
  - **Description**: The number of trees in the forest.
  - **Effect**: More trees can improve the model's performance but also increase computation time.
  - **Values Used**: `[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]` in grid search.

- **`max_depth`**: 
  - **Description**: The maximum depth of the tree.
  - **Effect**: Limits the number of levels in each decision tree to prevent overfitting.
  - **Values Used**: `[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` in grid search.

- **`min_samples_split`**: 
  - **Description**: The minimum number of samples required to split an internal node.
  - **Effect**: Higher values prevent the model from learning overly specific patterns.
  - **Values Used**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]` in grid search.

- **`min_samples_leaf`**: 
  - **Description**: The minimum number of samples required to be at a leaf node.
  - **Effect**: Controls the size of the tree; larger values simplify the model.
  - **Values Used**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]` in grid search.

- **`max_features`**: 
  - **Description**: The number of features to consider when looking for the best split.
  - **Options**: 
    - `'log2'`: Use log2 of the number of features.
    - `'sqrt'`: Use the square root of the number of features.
    - `None`: Use all features.
  - **Usage**: All options are explored in grid search.

- **`bootstrap`**: 
  - **Description**: Whether bootstrap samples are used when building trees.
  - **Options**: 
    - `True`: Use bootstrap samples.
    - `False`: Use the entire dataset.
  - **Usage**: Both `True` and `False` are considered.

- **`class_weight`**: 
  - **Description**: Weights associated with classes to handle class imbalance.
  - **Options**: 
    - `None`: All classes have weight one.
    - `'balanced'`: Adjusts weights inversely proportional to class frequencies.
  - **Usage**: Both `None` and `'balanced'` are considered.

The model is optimized using grid search with cross-validation to find the best combination of these parameters based on the F1 score.

# LightGBM Model Parameters

The LightGBM model in this project is implemented using the `LGBMClassifier` class from the LightGBM library. Below is a detailed explanation of the parameters used:

- **`objective`**: 
  - **Description**: Specifies the learning task and the corresponding objective function.
  - **Value Used**: `'binary'` for binary classification tasks.

- **`metric`**: 
  - **Description**: The metric used for evaluation.
  - **Value Used**: `'auc'` to evaluate the model using the Area Under the ROC Curve.

- **`boosting_type`**: 
  - **Description**: The type of boosting to use.
  - **Value Used**: `'gbdt'` which stands for Gradient Boosting Decision Tree.

- **`verbosity`**: 
  - **Description**: Controls the amount of information LightGBM outputs.
  - **Value Used**: `-1` to suppress all messages except for errors.

- **`seed`**: 
  - **Description**: The random seed for reproducibility.
  - **Value Used**: `42`.

### Hyperparameters Tuned via Grid Search

- **`n_estimators`**: 
  - **Description**: The number of boosting rounds.
  - **Values Used**: `[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]`.

- **`max_depth`**: 
  - **Description**: The maximum depth of a tree.
  - **Values Used**: `[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`.

- **`min_samples_split`**: 
  - **Description**: The minimum number of samples required to split an internal node.
  - **Values Used**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`.

- **`min_samples_leaf`**: 
  - **Description**: The minimum number of samples required to be at a leaf node.
  - **Values Used**: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]`.

- **`learning_rate`**: 
  - **Description**: The learning rate shrinks the contribution of each tree.
  - **Values Used**: `[0.01, 0.1, 0.2]`.

- **`lambda_l1`**: 
  - **Description**: L1 regularization term on weights.
  - **Values Used**: `[0, 0.1, 1]`.

- **`lambda_l2`**: 
  - **Description**: L2 regularization term on weights.
  - **Values Used**: `[0, 0.1, 1]`.

- **`scale_pos_weight`**: 
  - **Description**: Controls the balance of positive and negative weights.
  - **Values Used**: `[1, len(y)/sum(y)]`.

The model is optimized using grid search with cross-validation to find the best combination of these parameters based on the ROC AUC score.

