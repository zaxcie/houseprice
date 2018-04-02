import pandas as pd
import numpy as np
from src.configs import *
from src.features.utils import categorical_to_ordinal


def transform_quality_var_into_ordinal(workset):
    feature_names = list()
    feature_names_to_ignore = list()

    for col in QUALITY_COL:
        new_col_name = col + "_ordinal"
        workset[new_col_name] = categorical_to_ordinal(workset[col])
        feature_names.append(new_col_name)
        feature_names_to_ignore.append(col)

    return workset, feature_names, feature_names_to_ignore
