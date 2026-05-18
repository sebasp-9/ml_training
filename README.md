# Explicit pip installations
- pandas
- scikit-learn
- seaborn
- xgboost

# Notes
Initial run hit F1 of **0.5387741259669531**
- n_estimators=100
- max_depth=6
- learning_rate=0.3
- subsample=1

Small tuning recommended from guide hit F1 of **0.565038356311371**
- n_estimators=200
- max_depth=6
- learning_rate=0.1
- subsample=0.8

## Parameters
- `n_estimators`[🔗](https://xgboosting.com/configure-xgboost-n_estimators-parameter/): (recommended start = 100) number of trees (estimators) in the model, allowing you to control the model’s complexity and performance
  - start with a moderate value for n_estimators (e.g., 100) and adjust it based on the model’s performance and computational constraints
  - keep in mind that n_estimators interacts with other parameters, such as learning_rate, these parameters can be tuned together to achieve the best performance
- `max_depth`[🔗](https://xgboosting.com/configure-xgboost-max_depth-parameter/): (default = 6) maximum depth of a tree in the model, influence the model’s complexity and its ability to generalize
- `learning_rate`[🔗](https://xgboosting.com/configure-xgboost-learning_rate-parameter/): (0-1 inclusive, default = 0.3) controls the step size at each boosting iteration
- `subsample`[🔗](https://xgboosting.com/configure-xgboost-subsample-parameter/): (0-1 inclusive, default = 1) controls the fraction of observations used for each tree
- `scale_pos_weight`[🔗](https://xgboosting.com/xgboost-configure-scale_pos_weight-parameter/): calculate for imbalanced datasets

# TODO
1. [ ] Implement `scale_pos_weight`
2. [ ] Implement confusion matrix
3. [ ] Organize code
4. [X] Turn parameters into arguments for easy testing