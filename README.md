# OCT-H Models Overview

This paper divides OCT-H models into two main models based on solving methods: the heuristic method (refer to Section [Heuristic](#heuristic)) and global optimization based on warm start (refer to Section [Warm Start](#warm-start)).

## Parameters in OCT-H (Heuristic)

- **`max_depth`**: This parameter sets the maximum depth of the tree. A deeper tree can capture more complex patterns but may also lead to overfitting and more computation time. The default value is 2.

- **`α`**: This parameter controls the complexity of the tree. A larger value of `α` results in a sparser tree, which can help prevent overfitting. The default value is 0.01.

- **`N`**: This is the maximum number of features that can be used at a branch node. It also influences the complexity of the tree. The default value is 5.

- **`min_samples_split`**: This parameter specifies the minimum number of instances required at a branch node to consider splitting. The default value is 20.

- **`objective`**: This defines the objective of the OCT-H model.
  - `objective = accuracy`: The model aims to maximize accuracy (OCT-H-Acc).
  - `objective = F1-score`: The model aims to maximize the F1-score (OCT-H-F1).
  - `objective = cost-sensitive`: This is for the cost-sensitive OCT-H (CSOCT-H).
  
  For the credit scoring problem, the default value is `objective = F1-score`.

- **`C_FN`**: This parameter only functions when `objective = cost-sensitive` and is used to overcome class imbalance. The default value is the ratio of goods to bads.

- **`number_important_features`**: This parameter indicates the number of features selected by the Random Forest (RF) at each branch node. The default value is 15.

## Parameters in OCT-H (Optimal)

In the OCT-H (Optimal) model, the parameters generally have the same meanings as those in the OCT-H (Heuristic), with one key difference regarding the parameter `number_important_features`.

- **`number_important_features`**: In OCT-H (Optimal), this is the total number of features selected by the RF. Thus, the total number of features used by OCT-H (Optimal) cannot exceed `number_important_features`. The default value is 15.
