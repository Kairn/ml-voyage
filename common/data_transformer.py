"""
This module contains functions to be used for transforming data. The purpose is
to provide customized utilities for general project-wide use.
"""

import pandas as pd


def binary_prob_transform(series, threshold=0.5, inclusive=True):
    """
    Transforms probability values into binary decision values of 0 or 1 based
    on the specified threshold value. A value that is equal to the threshold
    value will be transformed into 1 or 0 when inclusive is True or False,
    respectively.
    :param series:      the input list
    :param threshold:   the positive desicion threshold
    :param inclusive:   whether the threshold is considered positive
    :return: the transformed list
    """

    new_series = []

    for s in series:
        if s > threshold:
            new_series.append(1)
        elif s < threshold:
            new_series.append(0)
        else:
            if inclusive:
                new_series.append(1)
            else:
                new_series.append(0)

    return new_series


def drop_list_outliers(data, feature, outliers):
    """
    Drops the outliers based on the query list and the feature, and then
    returns the dataframe without the outliers.
    :param data:        the source dataframe
    :param feature:     the column for querying the outliers
    :param outliers:    the list of outliers
    :return: the dataframe with outliers removed
    """

    new_data = pd.DataFrame(data)

    for ol in outliers:
        new_data.drop(new_data[new_data[feature] == ol].index, inplace=True)

    return new_data
