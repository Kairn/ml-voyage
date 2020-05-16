"""
This module contains functions to be used for testing the accuracy of a model
by comparing the predictions against the training or real results.
"""


def get_simple_accuracy(pred_series, val_series):
    """
    Compares the values from the prediction set to the values in the validation
    set. This is a simple boolean comparison and will return a percentage based
    on the number of correct values. Only works when values in both sets are
    ordered in the same way, or the result is undefined. Will return -1 if the
    lengths of the sets do not match.
    :param pred_series: the series of the prediction values
    :param val_series:  the series of the validation values
    :return: the accuracy percentage
    """

    if pred_series is None or val_series is None \
            or len(pred_series) != len(val_series):
        return -1

    total = len(val_series)
    correct = 0

    for p, v in zip(pred_series, val_series):
        if p == v:
            correct += 1

    return round(correct / total, 4)
