"""
This is a Linear Regression model built around several features and outputs the
average prediction values from all feature models. Features are selected based
on historical facts about the protocol during the incident.
* Most recent prediction score over real data: ~71%.
"""

import os
import math
import numpy
import pandas as pd
from sklearn.linear_model import LinearRegression

from common.accuracy_report import get_simple_accuracy
from common.data_transformer import binary_prob_transform, drop_list_outliers


def get_linear_model(data, feature):
    """
    Returns the linear regression model fitted by the specified feature in the
    given dataset.
    :param data:    the dataset/dataframe
    :param feature: the independent variable name
    :return: the fitted model ready for prediction
    """

    lr_model = LinearRegression()
    y = data.Survived
    x = numpy.array(data[feature])
    x = x.reshape(-1, 1)

    lr_model.fit(x, y)
    return lr_model


def get_multi_linear_result(data, feature_list, model_list):
    """
    Processes the dataset and predicts the outcome by averaging the values from
    all linear models. Missing values won't have any effect.
    :param data:            the test data
    :param feature_list:    the feature list
    :param model_list:      the models obtained for the features
    :return: the series represents the predictions
    """

    res = []

    for _, row in data.iterrows():
        total = 0
        for f in feature_list:
            val = row[f]
            if math.isnan(val):
                total += 0.5
            else:
                total += model_list[f].predict([[val]])[0]
        res.append(total / fet_count)

    return res


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
threshold = train_input_data.Survived.mean()
features = ['Pclass', 'Sex', 'Age']
fet_count = len(features)
outliers = [298, 339, 401, 405, 499]
models = {}

# Delete outliers and fit feature models
train_input_data = drop_list_outliers(train_input_data, pid_col, outliers)
train_ref = train_input_data[surv_col]
for fet in features:
    df = train_input_data[[surv_col, fet]]
    df = df.dropna()
    models[fet] = get_linear_model(df, fet)

# Predict the results with manual process
train_input_data = train_input_data[features]
train_input_data.fillna(numpy.NaN, inplace=True)
test_input_data = test_input_data[features]
test_input_data.fillna(numpy.NaN, inplace=True)

train_res = get_multi_linear_result(train_input_data, features, models)
surv_res = get_multi_linear_result(test_input_data, features, models)
train_res = binary_prob_transform(train_res, threshold, False)
surv_res = binary_prob_transform(surv_res, threshold, False)

# Report
accuracy = get_simple_accuracy(train_res, train_ref)
print('Training data accuracy: {0}%.'.format(accuracy * 100))

# Output
output_file_name = 'res_linear_regression.csv'
output_path = os.path.join(os.path.dirname(__file__), '..',
                           'results', output_file_name)

result = pd.DataFrame({pid_col: pids, surv_col: surv_res})
result.to_csv(output_path, index=False)
