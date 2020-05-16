"""
This is a Decision Tree model that uses branch decision making to predict the
outcome value based on one or several characteristics of the input. It is a
relatively simple model that performs well on large datasets. However, it has
the tendency to cause overfitting. This can be improved with data pruning and
refining.
"""

import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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
rand_state = 516122
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

# Narrow and refine data
narrow_data = train_input_data[features]
narrow_test = test_input_data[features]
reg_data = narrow_data.fillna(narrow_data.mean())
reg_res = train_input_data[surv_col]
narrow_test = narrow_test.fillna(narrow_test.mean())

# Perform regression
reg_model = DecisionTreeRegressor(random_state=rand_state)
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
output_file_name = 'res_decision_tree_model.csv'
output_path = os.path.join(os.path.dirname(__file__), '..',
                           'results', output_file_name)

result = pd.DataFrame({pid_col: pids, surv_col: surv_res})
result.to_csv(output_path, index=False)
