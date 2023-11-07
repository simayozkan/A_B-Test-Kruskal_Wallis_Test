import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import MultiComparison

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/dataset.csv")

df.head()

df.info()

df.isna().sum()

df.describe().T

df.groupby("Promotion").agg({"SalesInThousands": ["count", "mean"]})

df["Promotion"].value_counts()

#h0 : no statistically difference between promotion and sales in thousands
#h1 : statistically difference between promotion and sales in thousands

#p-values lower than 0.05 so reject null hypothesis, normality assumption doesn't hold

for group in list(df["Promotion"].unique()):
    pvalue = shapiro(df.loc[df["Promotion"] == group, "SalesInThousands"])[1]
    print(group, 'p-value: %.4f' % pvalue)

#There are more than two groups so we use Kruskal Wallıs test
kruskal(df.loc[df["Promotion"] == 1, "SalesInThousands"],
        df.loc[df["Promotion"] == 2, "SalesInThousands"],
        df.loc[df["Promotion"] == 3, "SalesInThousands"])


#Tukey's HSD
comparison = MultiComparison(df['SalesInThousands'], df['Promotion'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())



