import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# dpi: plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# dpi: sns
sns.set(rc={"figure.dpi":300, "savefig.dpi":300})
sns.set_context("notebook")
sns.set_style("ticks")

# data load
X_data = pd.read_csv("X.csv")
df_fa = pd.read_csv("data_fa.csv")
X_pca = pd.read_csv("data_pca.csv")
y = pd.read_csv("y.csv")

# Modeling
def bootstrap_shuffle_model(X, y, model_name,
                             n_boot=1000, random_state=42):
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)
    n = len(y)
    scores = np.empty(n_boot)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    lst_coef = []
    for i in range(n_boot):
        if i % 100 == 0: print(i)
        
        perm_idx = rng.permutation(n)
        X_shuff, y_shuff = X_arr[perm_idx], y_arr[perm_idx]
        X_shuff[0]
        X_train, X_test, y_train, y_test = train_test_split(X_shuff, y_shuff,
                                                            test_size=0.2,
                                                            random_state=42
                                                            )
        model = models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[i] = mean_squared_error(y_test, y_pred)

        if model_name in ['lr', 'ridge', 'lasso']:
            lst_coef.append(model.coef_)
        else:
            lst_coef.append(model.feature_importances_)
    
    return scores, lst_coef

x_dict = {"raw": X_data.copy(),
          "fa": df_fa.copy(),
          "pca": X_pca.copy()}

models = {
    "lr": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "rf": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42
        ),

    "xgb": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
        )
    }

ml_result = []
for x_name in x_dict.keys():
    print(x_name)
    for model_name in models.keys():
        print(model_name)
        scores, lst_coef = bootstrap_shuffle_model(x_dict[x_name], y, model_name,
                                     n_boot=1000, random_state=42)
        ml_result.append([x_name,
                          model_name,
                          scores,
                          lst_coef
                          ])

ml_res_df = pd.DataFrame(ml_result,
                         columns=["data", "model", "MSE", "coef"]
                         )
coef = pd.DataFrame(ml_res_df["coef"].tolist(), index=[f"{x_name}_{model_name}" for x_name in x_dict.keys() for model_name in models.keys()])

for i in range(1000):
    for j in range(15):
        if coef[i][j].ndim==2:
            coef[i][j] = coef[i][j].reshape(-1,)

coef = coef.T
coef.to_csv("R_coef.csv", index=None)

# 95% Confidence Interval
df_quantile = pd.DataFrame()
df_sign = pd.DataFrame()
for col in coef.columns:
    df_coef_temp = pd.DataFrame(coef[col].tolist())
    lower_q = df_coef_temp.quantile(0.025)
    upper_q = df_coef_temp.quantile(0.975)
    
    mask = (df_coef_temp >= lower_q) & (df_coef_temp <= upper_q)
    df_clean = df_coef_temp.where(mask)
    
    df_clean.to_csv(f"95p_{col}.csv", index=None)
    df_quantile = pd.concat([df_quantile, pd.DataFrame([lower_q, upper_q]).T.apply(tuple, axis=1)], axis=1)
    
    a = pd.concat([lower_q, upper_q], axis=1)
    df_sign = pd.concat([df_sign, np.abs(a).sum(axis=1) == np.abs(a.sum(axis=1))], axis=1)

df_quantile.columns = coef.columns
raw_quantile = df_quantile[[f"raw_{model_name}" for model_name in models.keys()]]
raw_quantile.to_csv("quantile_raw.csv", index=None)
fa_quantile = df_quantile[[f"fa_{model_name}" for model_name in models.keys()]]
fa_quantile.dropna(inplace=True)
fa_quantile.to_csv("quantile_fa.csv", index=None)
pca_quantile = df_quantile[[f"pca_{model_name}" for model_name in models.keys()]]
pca_quantile.dropna(inplace=True)
pca_quantile.to_csv("quantile_pca.csv", index=None)

df_sign.columns = coef.columns
raw_sign = df_sign[[f"raw_{model_name}" for model_name in models.keys()]]
raw_sign.to_csv("sign_raw.csv", index=None)
fa_sign = df_sign[[f"fa_{model_name}" for model_name in models.keys()]]
fa_sign.dropna(inplace=True)
fa_sign.to_csv("sign_fa.csv", index=None)
pca_sign = df_sign[[f"pca_{model_name}" for model_name in models.keys()]]
pca_sign.dropna(inplace=True)
pca_sign.to_csv("sign_pca.csv", index=None)

# MSE
df_mse = pd.DataFrame(ml_res_df["MSE"].tolist()).T
df_mse.columns = coef.columns
df_mse

for col in coef.columns:
    df_mse_temp = pd.DataFrame(df_mse[col].tolist())
    lower_q = df_mse_temp.quantile(0.025)
    upper_q = df_mse_temp.quantile(0.975)
    
    mask = (df_mse_temp >= lower_q) & (df_mse_temp <= upper_q)
    df_clean = df_mse_temp.where(mask)
    df_clean.dropna(inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    df_clean.to_csv(f"mse_{col}.csv", index=None)
    print(round(lower_q[0], 5), round(upper_q[0], 5), round(df_clean.mean()[0], 5))

mse_mean = df_mse.mean().apply(round, ndigits=5)
mse_mean.to_csv("mse_mean.csv")
