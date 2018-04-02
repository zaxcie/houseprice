import pandas as pd
import numpy as np
from src.configs import QUALITY_ORDINAL_MAPPING


def categorical_to_ordinal(series, mapping=QUALITY_ORDINAL_MAPPING):
    return series.map(mapping)
