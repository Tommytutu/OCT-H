# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:16:52 2024

@author: JC TU
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
import LBC
import dataset
import OCTH_warm_start

timelimit = 600
alpha = 0.01

# Load dataset
X, y = dataset.loadData('UK')

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Define parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': [10, 30, 50],
    'class_weight': [None, 'balanced']
}

# Initialize RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=4, n_jobs=-1, verbose=2, return_train_score=True)
grid_search.fit(X, y)

# Output the best parameter combination
print("Best parameter combination:", grid_search.best_params_)

# Train the model with the best parameter combination
best_model = grid_search.best_estimator_

# Get feature importance ranking
feature_importance = best_model.feature_importances_

# Number of important features to select
number_important_features_list = [10, 15, 20]

# Optimal classification tree depth
depth_list = [1, 2, 3]

ax = {}
bx = {}

for number_important_features in number_important_features_list:
    top_indices = np.argsort(feature_importance)[::-1][:number_important_features]
    x_train1 = x_train[:, top_indices]

    for d in depth_list:
        if d <= 1:
            model = LBC.LBC(timelimit=300, objective='F1-score', alpha=alpha, N=5)
            model.fit(x_train1, y_train)
            bx[1] = model.b
            for f in range(x_train1.shape[1]):
                ax[f, 1] = model.a[f]
        else:
            model = OCTH_warm_start.OCTH_warm_start(max_depth=d, alpha=alpha, N=5, objective='F1-score', warmstart=True, timelimit=timelimit, ax=ax, bx=bx, output=True)
            model.fit(x_train1, y_train)
            ax = model.a
            bx = model.b

        y_test_pre = model.predict(x_test[:, top_indices])
        f1_test = f1_score(y_test, y_test_pre)