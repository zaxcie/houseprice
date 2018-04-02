library(GGally)

data = read.csv("../data/raw/train.csv")
cols_with_na = c("BsmtQual", "BsmtCond", "BsmtExposure",
                 "BsmtFinType1", "BsmtFinType2", "GarageType",
                 "GarageFinish", "GarageQual", "GarageCond",
                 "PoolQC", "Fence", "MiscFeature", "Alley")



ggpairs(data, cardinality_threshold=25)
