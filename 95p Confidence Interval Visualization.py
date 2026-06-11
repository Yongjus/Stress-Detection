import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

def name_change(x):
    return x.replace("lr", "Linear Regression").replace("lasso", "Lasso").replace("ridge", "Ridge").replace("rf", "Random Forest").replace("xgb", "XGBoost")

file_names = os.listdir()
file_names = list(filter(lambda x: x.startswith("95p_"), file_names))
real_names = list(map(name_change, file_names))

for file_name, real_name in zip(file_names, real_names):
    data = pd.read_csv(file_name)
    max_data = data.max().values
    min_data = data.min().values
    mean_data = data.mean().values
    
    errorbar_data = pd.DataFrame([mean_data, abs(min_data-mean_data), abs(max_data-mean_data)]).T
    errorbar_data.columns = ["y", "yerr0", "yerr1"]
    errorbar_data = errorbar_data.sort_values("y", ascending=False)
        
    plt.errorbar(errorbar_data.index.astype("str"), 
                 errorbar_data["y"], 
                 yerr=[errorbar_data["yerr0"], errorbar_data["yerr1"]],
                 fmt=".",
                 color = 'r',
                 capsize=6,
                 ecolor='grey')
    
    plt.xlabel("feature numbers")
    plt.ylabel("coefficient")
    
    if real_name.split("_")[1] == "raw":
        plt.xticks(rotation=90, fontsize=5)
        title = f"Data: Raw Data\nModel: {real_name.split('_')[2].split('.')[0]}"
    else:
        plt.xticks(rotation=90)
        title = f"Data: Applying {real_name.split('_')[1].upper()}\nModel: {real_name.split('_')[2].split('.')[0]}"
    plt.xticks(rotation=90)
    title = "Coefficient with 95% Bootstrap Confidence Interval:\nCombination of FA and Lasso"
    plt.title(title)
    plt.savefig(fr"plot\{title.replace("\n", " ").replace(":", "")}.png")
    plt.show()
