"""
This is a random model which predicts the passenger's survival randomly.
This model is not practical or useful, and it is meant to be provide random
sample predictions.
"""

import os
import random
import pandas as pd

random.seed()

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
surv_res = []

# Generate random results
for i in range(len(pids)):
    surv_res.append(random.randint(0, 1))

output_file_name = 'res_random_model.csv'
output_path = os.path.join(os.path.dirname(__file__), '..',
                           'results', output_file_name)

result = pd.DataFrame({pid_col: pids, surv_col: surv_res})
result.to_csv(output_path, index=False)
