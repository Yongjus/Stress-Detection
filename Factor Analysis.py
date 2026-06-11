# Import modules
!pip install factor_analyzer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# dpi: plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# dpi: sns
sns.set(rc={"figure.dpi":300, "savefig.dpi":300})
sns.set_context("notebook")
sns.set_style("ticks")

# data load
df = pd.read_csv("hundred_data.csv")
df.drop([73, 19, 61, 31], inplace=True)
df = df.reset_index(drop=True)
df.drop([f"f{i+1}" for i in range(4)], axis=1, inplace=True)

# seperate
data_sacq = df[[f"sacq{i+1}" for i in range(25)]] /20
data_sleep = df[[f"sleep{i+1}" for i in range(2)]] /20
data_psqi = df[[f"psqi{i+1}" for i in range(9)]] /25
data_cesd = df[[f"cesd{i+1}" for i in range(20)]] /25
data_happiness = df[[f"shs{i+1}" for i in range(2)]
                    + [f"swls{i+1}" for i in range(5)]] /20
data_pss = df[[f"pss{i+1}" for i in range(10)]] /20

datas = [data_sacq.astype(int),
        data_sleep.astype(int),
        data_psqi.astype(int),
        data_cesd.astype(int),
        data_happiness.astype(int)
        ]

data_names = ['sacq', 'sleep', 'psqi', 'cesd', 'happiness']

X = pd.concat(datas, axis=1)
y = df[df.columns[df.columns.str.startswith("pss")]].mean(axis=1)

def bartlett_test(df):
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
 
    return chi_square_value, p_value

def KMOtest(df):
    from factor_analyzer.factor_analyzer import calculate_kmo

    _, kmo_model = calculate_kmo(df)

    return kmo_model

def factor_analysis(df, n_factors=None, method=None):
    from factor_analyzer import FactorAnalyzer
            
    if n_factors:
        fa = FactorAnalyzer(n_factors=n_factors, rotation=method)
        fa.fit(df)
        
    else:
        fa = FactorAnalyzer(n_factors=len(df), rotation=method)
        fa.fit(df)
        
        # Kaiser's Rule
        ev, _ = fa.get_eigenvalues() # print(ev)
        ev_1 = [i for i in ev if i >= 1] # ev >= 1
        print(ev_1)
        print(len(ev_1))
        return ev, len(ev_1)
    
    loadings = pd.DataFrame(fa.loadings_, index=df.columns)
    
    fa.get_factor_variance()
    variance = pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'])

    return loadings, variance

def plot_fa_heatmap(loadings, name):
    plt.figure(figsize=(15,20))
    sns.heatmap(loadings,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1)
    plt.title(f"Factor Analysis Heatmap: {name}")
    plt.savefig(f"{name}_fa_heatmap.png")
    plt.show()

# Factor Analysis
fa_test = []
fa_loading_df = pd.DataFrame([])
fa_variance = []
df_fa = pd.DataFrame([])

for name, data, n in zip(data_names, datas, [8, 1, 3, 5, 2]):
    print(name)
    b, b_p = bartlett_test(data)
    print("bartlett", b, b_p)
    
    KMO = KMOtest(data)
    print("KMO", KMO)
    
    fa_test.append([b, b_p, KMO])
    
    loadings, variance = factor_analysis(data, n_factors=n, method="varimax")
    loadings = loadings[abs(loadings) > 0.3].sort_values([i for i in range(n)], ascending=False)
    loadings.columns = [f"factor {i}" for i in range(n)]
    loadings.to_csv(f"r_fa_{name}_loadings.csv")
    plot_fa_heatmap(loadings, name)
    
    if n <= 1: continue
    for fac in [f"factor {i}" for i in range(n)]:
        a = data[loadings[~loadings[fac].isna()].index] * loadings[fac].dropna()
        df_fa = pd.concat([df_fa, a.sum(axis=1)], axis=1)

df_fa.columns = [f"factor {i}" for i in range(len(df_fa.columns))]
df_fa.to_csv("data_fa.csv", index=None)
