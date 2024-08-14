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

python
import LBC
import OCTH_warm_start
model = OCTH_warm_start.OCTH_warm_start(max_depth=2, alpha=0.01, N=5, objective='F1-score', warmstart=True, output=True)
model.fit(x_train1, y_train)


# Logistic Regression Model Parameters

This section provides an overview of the parameters used in the Logistic Regression model for credit scoring.

- **`C`**: Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization. The default values used in the grid search are `[0.1, 1, 10]`.

- **`penalty`**: Specifies the norm used in the penalization. The options are `'l1'` and `'l2'`. `'l1'` can lead to sparse solutions, where some coefficients are exactly zero, which can be useful for feature selection.

- **`class_weight`**: Weights associated with classes. If not given, all classes are supposed to have weight one. The `'balanced'` mode uses the values of `y` to automatically adjust weights inversely proportional to class frequencies in the input data.

- **`solver`**: Algorithm to use in the optimization problem. In this code, `'saga'` is used, which supports both `l1` and `l2` penalties and is suitable for large datasets.




