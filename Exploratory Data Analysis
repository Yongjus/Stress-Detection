# Import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dpi: plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Load data
df = pd.read_csv(r"data.csv", encoding="cp949")
df.drop([73, 19, 61, 31], inplace=True)
df = df.reset_index(drop=True)
data = pd.DataFrame()

# Labelling
label_major = {"인문계열": "Humanities",
              "사회계열": "Social Sciences",
              "자연계열": "Natural Sciences",
              "공학계열": "Engineering",
              "예체능계열": "Arts and\nPhysical Education",
              "의약계열": "Medical and\nHealth Sciences"}

data["major"] = df["P3"].map(label_major)

label_sex = {"남성": "Male",
            "여성": "Female"}

data["sex"] = df["P4"].map(label_sex)

label_grade = {"1학년": "freshman",
              "2학년": "sophomore",
              "3학년": "junior",
              "4학년": "senior"}

data["grade"] = df["P5"].map(label_grade)

label_recess = {"없음": "No", "있음": "Yes"}

data["recess"] = df["P6"].map(label_recess)

# EDA
color = [f"C{i}" for i in range(10)]

for col in data.columns:
    plt.figure(figsize=(12,8))
    a = data[col].value_counts()
    plt.subplot(1,2,1)
    plt.barh(a.index, a, 
             color=color[:len(a)]
             )
    for i in range(len(a)):
        plt.text(a.iloc[i]/2, i, str(a.iloc[i]), 
                 fontsize=15,
                 fontweight="bold"
                 )
    plt.yticks(fontsize=10, fontweight="bold")
    
    plt.subplot(1,2,2)
    plt.pie(a,
            wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5},
            autopct='%.2f%%',
            textprops={"size": 12, "weight": "bold"}
            )
    
    plt.suptitle(col.capitalize(), fontsize=20, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(f"{col} frequency analysis.png")
    plt.show()
