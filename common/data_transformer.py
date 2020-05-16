"""
This module contains functions to be used for transforming data. The purpose is
to provide customized utilities for general project-wide use.
"""


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
