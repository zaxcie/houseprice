import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import chi2
from scipy.stats.stats import pearsonr
import statsmodels.api as sm


def linear_relationship_between_target_and_ordinal_var(ordinal, target, ordinal_name, cv=10):
    X = sm.add_constant(ordinal.astype(float))
    est = sm.OLS(target, X)
    est2 = est.fit()
    predict_sm = est2.predict(X)
    print(est2.summary())

    fig, ax = plt.subplots()
    plt.scatter(ordinal, target, color='black')
    plt.plot(ordinal, predict_sm, color='blue', linewidth=3)
    ax.set_xlabel(ordinal_name)
    ax.set_ylabel('Target')
    plt.show()

    pearson_coefs = pearsonr(ordinal, target)
    print("Pearson coef: " + str(pearson_coefs[0]) + " with a p-value of " + str(pearson_coefs[1]))
    print(est2.rsquared)

    print("-----------------------------")
    print("\n\n\n")

    return est2.rsquared, pearson_coefs[0]


def multilinear_relationship_between_target_and_ordinal_var(ordinal, target):
    X = sm.add_constant(ordinal.astype(float))
    est = sm.OLS(target, X)
    est2 = est.fit()
    print(est2.summary())

    print("-----------------------------")
    print("\n\n\n")