"""
Econ 406
Homework 5
"""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
#exercise1 Wage
def first_exercise():
    """
    generate all the output to understand the impact of different variables
    on expected wage rate.
    Returns
    -------
    None.

    """
#1.1 load the data to make sure ready for analysis
    dataset = pd.read_csv("wage.csv")
    dataset = dataset.dropna()

#1.2 data visualization
    plt.scatter(dataset['educ'], dataset['wage'])
    plt.xlabel("education")
    plt.ylabel("wages")
    plt.scatter(dataset['exper'], dataset['wage'])
    plt.xlabel("experience")
    plt.ylabel("wages")
    plt.scatter(dataset['tenure'], dataset['wage'])
    plt.xlabel("tenure")
    plt.ylabel("wages")

#1.3 OLS or Logistic regression
    sns.lmplot(x='educ', y='wage', data=dataset)
    sns.lmplot(x='exper', y='wage', data=dataset)
    sns.lmplot(x='tenure', y='wage', data=dataset)
# As we see that the regression line never goes below 0, so there is no
# need to use logistic regression for this. Besides, the logit model is better
# to use when have multiple possible outcomes.Moreover OLS should be better
# here as it shows a linear relationship which is more direct.

#1.4 data generating process for wages
#1.5 generate the regression table
    df_predict = dataset[['wage', 'educ', 'tenure']]
    mod = smf.ols(formula='wage~educ+tenure', data=df_predict)
    res = mod.fit()
    print(res.summary())
# as we see from the regression line that there is no clear correlations
# between the wages and experience. Thus the proposed modle only employs
# years of educations and tenures.

#1.6 there is no p value greater than 0.05.
# which means it's less than 0.05 percent chance we are wrong to reject that
# there is no correlations of education level and tenure regarding to wages.
# which means that both the education level and tenure are highly correlated
# to the wages.

#1.7 The R-squared is 0.302 means there is 30.2% of the data fit the
# regression modle, and it helps to explain how well the modle of prediction

#1.8
    hypo = pd.DataFrame({'wage': [150], 'educ': [170], 'tenure': [293]})
    res.predict(hypo)
# when educ equals 170 and tenure equals 293, the hourly wages is expect to
# be 150.

#exercise2: Diabetes
def second_exercise():
    """
    predict whether or not a patient has diabetes, based on certain diagnosis
    measurements included in the dataset.

    Returns
    -------
    None.

    """
#2.1 prep the data
    dataset = pd.read_csv("diabetes.csv")
    dataset = dataset.dropna()
    dataset['diabetes'] = dataset['diabetes'].replace('neg', 0)
    dataset['diabetes'] = dataset['diabetes'].replace('pos', 1)

#2.2 data visualization
    sns.lmplot(x="pedigree", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="pregnant", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="pressure", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="triceps", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="insulin", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="mass", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="mass", y="diabetes", data=dataset, logistic=True,
               y_jitter=0.03)
    sns.lmplot(x="pedigree", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="pedigree", y="diabetes", data=dataset, logistic=True,
               y_jitter=0.03)
    sns.lmplot(x="glucose", y="diabetes", data=dataset, y_jitter=0.03)
    sns.lmplot(x="glucose", y="diabetes", data=dataset, logistic=True,
               y_jitter=0.03)

#2.3 Logistic regression is more suitable here since we are analyzing the
# probability, and we can't have a negative probability or greater than one.
# so we want to make sure the regression line is between zero and one
# otherwise it's meaningless with values exceeds the boundary, thus we prefer
# Logistic regression here.

#2.4 data genrating model for diabetes
    df_rhs = dataset[['pedigree', 'glucose', 'mass']]
    df_rhs = sm.add_constant(df_rhs)
    df_lhs = dataset['diabetes']
    logit_mod = sm.Logit(df_lhs, df_rhs)
    logit_res = logit_mod.fit()
    print(logit_res.summary())

#2.5 estimate the model
#the p values for pedigree. glucose, mass are all below 0.05 which means these
#three terms are all shows strong correlation with diabetes. the pseudo R-squ
#is about 27% wich means 27% can be explained by this model.

#2.6 the coefficient of pedigree is 1.17, of glucose is 0.04, of mass is 0.06.
#all of those coefficients are positive which means they are all positively
# and directly correlated to diabetes. Of these, the coefficient of pedigree
# is the biggest which means increase in one unit of pedigree leads to more
# than one unit change in the possibility of getting diabetes, thus it's the
# most influential independent variable.

#2.7 print out difference of the possibility for patient with 50th percentile
# and 75th and 25th percentile regarding the independent variables.
    percent_25 = logit_res.predict([1, np.percentile(dataset['pedigree'], 25),
                                    np.percentile(dataset['glucose'], 25),
                                    np.percentile(dataset['mass'], 25)])
    percent_50 = logit_res.predict([1, np.percentile(dataset['pedigree'], 50),
                                    np.percentile(dataset['glucose'], 50),
                                    np.percentile(dataset['mass'], 50)])
    percent_75 = logit_res.predict([1, np.percentile(dataset['pedigree'], 75),
                                    np.percentile(dataset['glucose'], 75),
                                    np.percentile(dataset['mass'], 75)])
    diff_75_and_50 = percent_75 - percent_50
    print(diff_75_and_50)
    diff_25_and_50 = percent_50 - percent_25
    print(diff_25_and_50)
