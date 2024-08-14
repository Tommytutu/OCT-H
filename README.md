# OCT-H 

Code for the paper *"Inherently Interpretable Machine Learning for Credit Scoring: Optimal Classification Tree with Hyperplane Splits"*.

The code will run in python3 and require [Gurobi11.0](https://www.gurobi.com/products/gurobi-optimizer/) solver.

run_example.py is the 

## Parameters 

- **`max_depth`**: This parameter sets the maximum depth of the tree. A deeper tree can capture more complex patterns but may also lead to overfitting and more computation time. The default value is 2.

- **'alpha'**: This parameter controls the complexity of the tree. A larger value of `α` results in a sparser tree, which can help prevent overfitting. The default value is 0.01.

- **`N`**: This is the maximum number of features that can be used at a branch node. It also influences the complexity of the tree. The default value is 5.

- **`min_samples_split`**: This parameter specifies the minimum number of instances required at a branch node to consider splitting. The default value is 20.

- **`objective`**: This defines the objective of the OCT-H model.
  - `objective = accuracy`: The model aims to maximize accuracy (OCT-H-Acc).
  - `objective = F1-score`: The model aims to maximize the F1-score (OCT-H-F1).
  - `objective = cost-sensitive`: This is for the cost-sensitive OCT-H (CSOCT-H).
  
  For the credit scoring problem, the default value is `objective = F1-score`.

- **`CFN`**: This parameter only functions when `objective = cost-sensitive` and is used to overcome class imbalance. The default value is the ratio of goods to bads.

- **`number_important_features`**: This parameter indicates the number of features selected by the Random Forest (RF) at each branch node. The default value is 15.


## How to run the code

* **Step 1**: Detect the violations of index-exchangeability condition

  Apply the function **[NV,violation_index_equal, violation_index_inequal]=NV_index_exchangeability(A)**, we get $NV=54$. Thus, $\mathbf{A}$ violates the index-exchangeability condition.
  In such cases, we have two options: directly derive the priority vector by the function **MNVLLSM**, goes to Step 2; communicate with the DM and use function **NPRAOC** to provide some
  modification suggestions and get more coherent preferences, goes to Step 3.

  ```matlab
  clear;
  clc;
  A=[1    	5    	3    	7    	6    	6    	 1/3	 1/4
   1/5	1    	 1/3	5    	3    	3    	 1/5	 1/7
   1/3	3    	1    	6    	3    	4    	6    	 1/5
   1/7	 1/5	 1/6	1    	 1/3	 1/4	 1/7	 1/8
   1/6	 1/3	 1/3	3    	1    	 1/2	 1/5	 1/6
   1/6	 1/3	 1/4	4    	2    	1    	 1/5	 1/6
   3    	5    	 1/6	7    	5    	5    	1    	 1/2
   4    	7    	5    	8    	6    	6    	2    	1    ];


  [NV,violation_index_equal, violation_index_inequal]=NV_index_exchangeability(A);


