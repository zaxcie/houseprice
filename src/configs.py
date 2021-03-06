COL_WITH_NAN_SIGNIFICATION = ["BsmtQual", "BsmtCond", "BsmtExposure",
                              "BsmtFinType1", "BsmtFinType2", "GarageType",
                              "GarageFinish", "GarageQual", "GarageCond",
                              "PoolQC", "Fence", "MiscFeature", "Alley"]

CONTINUOUS_COL = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                  "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                  "HalfBath", "GarageYrBlt", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                  "ScreenPorch", "PoolArea", "MiscVal", "YrSold"]

DISCRETE_COL = ["OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
                "GarageCars", "KitchenAbvGr"]

CAT_COL = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
           "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle",
           "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
           "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",
           "CentralAir", "Electrical", "Functional", "GarageType", "PavedDrive", "Fence", "MiscFeature",
           "MoSold", "SaleType", "SaleCondition"]

QUALITY_COL = ["FireplaceQu", "ExterQual", "GarageQual", "GarageCond", "PoolQC", "BsmtQual", "BsmtCond", "HeatingQC",
               "KitchenQual", "ExterCond"]

FINISH_COL = ["GarageFinish"]

QUALITY_ORDINAL_MAPPING = {'NO': 0,
                           'Po': 1,
                           'Fa': 2,
                           'TA': 3,
                           'Gd': 4,
                           'Ex': 5,
                           'MISSING': 0}

ORIGINAL_FEATURE_COLS = ['Id',
                         'MSSubClass',
                         'MSZoning',
                         'LotFrontage',
                         'LotArea',
                         'Street',
                         'Alley',
                         'LotShape',
                         'LandContour',
                         'Utilities',
                         'LotConfig',
                         'LandSlope',
                         'Neighborhood',
                         'Condition1',
                         'Condition2',
                         'BldgType',
                         'HouseStyle',
                         'OverallQual',
                         'OverallCond',
                         'YearBuilt',
                         'YearRemodAdd',
                         'RoofStyle',
                         'RoofMatl',
                         'Exterior1st',
                         'Exterior2nd',
                         'MasVnrType',
                         'MasVnrArea',
                         'ExterQual',
                         'ExterCond',
                         'Foundation',
                         'BsmtQual',
                         'BsmtCond',
                         'BsmtExposure',
                         'BsmtFinType1',
                         'BsmtFinSF1',
                         'BsmtFinType2',
                         'BsmtFinSF2',
                         'BsmtUnfSF',
                         'TotalBsmtSF',
                         'Heating',
                         'HeatingQC',
                         'CentralAir',
                         'Electrical',
                         '1stFlrSF',
                         '2ndFlrSF',
                         'LowQualFinSF',
                         'GrLivArea',
                         'BsmtFullBath',
                         'BsmtHalfBath',
                         'FullBath',
                         'HalfBath',
                         'BedroomAbvGr',
                         'KitchenAbvGr',
                         'KitchenQual',
                         'TotRmsAbvGrd',
                         'Functional',
                         'Fireplaces',
                         'FireplaceQu',
                         'GarageType',
                         'GarageYrBlt',
                         'GarageFinish',
                         'GarageCars',
                         'GarageArea',
                         'GarageQual',
                         'GarageCond',
                         'PavedDrive',
                         'WoodDeckSF',
                         'OpenPorchSF',
                         'EnclosedPorch',
                         '3SsnPorch',
                         'ScreenPorch',
                         'PoolArea',
                         'PoolQC',
                         'Fence',
                         'MiscFeature',
                         'MiscVal',
                         'MoSold',
                         'YrSold',
                         'SaleType',
                         'SaleCondition']