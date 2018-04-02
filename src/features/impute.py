import pandas as pd
import numpy as np


def impute_var_mean(series):
    try:
        index = series[(series != 'NO') & (series != 'MISSING')].index
    except Exception as e:
        index = series.index

    mean = series.iloc[index].mean()

    series = series.replace("MISSING", mean)
    series = series.replace("NO", 0)

    return series