"""
This is a Random Forest model which constructs multiple decision trees by
splitting the data into groups of sub-samples and outputs the average of
prediction values from all trees. Such methodology helps reduce the problem of
overfitting caused by relying on only one decision tree.
* Most recent prediction score over real data: ~74%.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common.accuracy_report import get_simple_accuracy
from common.data_transformer import binary_prob_transform

train_file_name = 'train.csv'
test_file_name = 'test.csv'
train_input_path = os.path.join(os.path.dirname(__file__), '..',
                                'datasets', train_file_name)
test_input_path = os.path.join(os.path.dirname(__file__), '..',
                               'datasets', test_file_name)
train_input_data = pd.read_csv(train_input_path)
test_input_data = pd.read_csv(test_input_path)

pid_col = train_input_data.columns[0]
surv_col = train_input_data.columns[1]

pids = test_input_data[pid_col]

# Convert sex category into boolean integers
train_input_data.loc[train_input_data.Sex == 'female', 'Sex'] = 0
train_input_data.loc[train_input_data.Sex == 'male', 'Sex'] = 1
test_input_data.loc[test_input_data.Sex == 'female', 'Sex'] = 0
test_input_data.loc[test_input_data.Sex == 'male', 'Sex'] = 1

# Declare the features to be used for regression
rand_state = 5171239
trees = 250
features = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Narrow and refine data
narrow_data = train_input_data[features]
narrow_test = test_input_data[features]
reg_data = narrow_data.fillna(narrow_data.mean())
reg_res = train_input_data[surv_col]
narrow_test = narrow_test.fillna(narrow_test.mean())

# Perform regression
reg_model = RandomForestRegressor(n_estimators=trees, random_state=rand_state)
reg_model.fit(reg_data, reg_res)
train_res = reg_model.predict(reg_data)
surv_res = reg_model.predict(narrow_test)

# Transform prediction numerals into binary integers
train_res = binary_prob_transform(train_res)
surv_res = binary_prob_transform(surv_res)

# Report
accuracy = get_simple_accuracy(train_res, train_input_data[surv_col])
print('Training data accuracy: {0}%.'.format(accuracy * 100))

# Output
output_file_name = 'res_random_forest_model.csv'
output_path = os.path.join(os.path.dirname(__file__), '..',
                           'results', output_file_name)

result = pd.DataFrame({pid_col: pids, surv_col: surv_res})
result.to_csv(output_path, index=False)
