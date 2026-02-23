# Import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dpi: plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# dpi: sns
sns.set(rc={"figure.dpi":300, "savefig.dpi":300})
sns.set_context("notebook")
sns.set_style("ticks")

# data load
df_sacq = pd.read_csv('data_sacq.csv')
df_sleep = pd.read_csv('data_sleep.csv')
df_psqi = pd.read_csv('data_psqi.csv')
df_cesd = pd.read_csv('data_cesd.csv')
df_happiness = pd.read_csv('data_happiness.csv')
df_pss = pd.read_csv('data_pss.csv')
df_sum = pd.read_csv("sum_data.csv")
df = pd.read_csv("data.csv", encoding="cp949")

data = pd.DataFrame()

# Labelling
label_major = {"인문계열": "Humanities",
              "사회계열": "Social Sciences",
              "교육계열": "Education",
              "자연계열": "Natural Sciences",
              "공학계열": "Engineering",
              "예체능계열": "Arts and Physical Education",
              "의약계열": "Medical and Health Sciences"}

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

user = pd.read_csv("user.csv")
user.drop([73, 19, 61, 31], inplace=True)
user.reset_index(drop=True, inplace=True)

df = pd.read_csv("hundred_data.csv")
df.drop([73, 19, 61, 31], inplace=True)
df.reset_index(drop=True, inplace=True)

# seperate
data_sacq = df[[f"sacq{i+1}" for i in range(25)]]
data_sleep = df[[f"sleep{i+1}" for i in range(2)]]
data_psqi = df[[f"psqi{i+1}" for i in range(9)]]
data_cesd = df[[f"cesd{i+1}" for i in range(20)]]
data_happiness = df[[f"shs{i+1}" for i in range(2)]
                    + [f"swls{i+1}" for i in range(5)]]
data_pss = df[[f"pss{i+1}" for i in range(10)]]

datas = [data_sacq,
        data_sleep,
        data_psqi,
        data_cesd,
        data_happiness,
        data_pss
        ]

data_names = ['sacq', 
              'sleep', 'psqi', 'cesd', 'happiness', 'pss']
group_cols = ['major', 'sex', 'grade', 'recess']

# Cronbach Alpha
def cronbach_alpha(df):
    """
    Compute Cronbach's Alpha
    """
    k = df.shape[1]
    item_var = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    
    return (k / (k - 1)) * (1 - (item_var.sum() / total_var))

cronbach_alpha_results = []

for name, data in zip(data_names, datas):
    cronbach_alpha_results.append([name, np.round(cronbach_alpha(data),3)])

cronbach_alpha_results.append(["Total", np.round(cronbach_alpha(pd.concat(datas, axis=1)),3)])
cronbach_alpha_df = pd.DataFrame(cronbach_alpha_results,
                                 columns=["indices", "alpha"])

print(cronbach_alpha_df)

cronbach_alpha_df.to_csv("r_cronbach_alpha.csv", index=None)

# describe stats indices
describe_indices = []
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)
    describe_indices.append(np.round(data.describe(), 3))
describe_indices_df = pd.DataFrame(describe_indices, index=data_names)
describe_indices_df.to_csv("r_describe_indices.csv")

# describe stats
describe_results = []
data = data_pss.mean(axis=1)
describe_df = pd.DataFrame()
for col in group_cols:
    for i in range(max(user[col].unique())+1):
        data = data_pss.mean(axis=1)
        if len(data[user[col] == i]) <=0: continue
        describe_results.append([col, i])
        describe_df = pd.concat([describe_df, data[user[col] == i].describe()], axis=1)
        
groupby_describe_df = pd.concat([pd.DataFrame(describe_results, columns=["group", "kind"]), describe_df.T.reset_index(drop=True)], axis=1)

print(groupby_describe_df)

groupby_describe_df.to_csv("r_groupby_describe.csv", index=None)

# kdeplot
# major
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)
    for i in [0, 1, 3, 4, 5]:
        sns.kdeplot(data[user["major"] == i])
    
    plt.legend(['Humanities', 'Social Sciences', 'Natural Sciences', 'Engineering', 'Arts and\nPhysical Education'])
    plt.title(name)
    plt.show()

# sex
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)
    for i in range(2):
        sns.kdeplot(data[user["sex"] == i])
    plt.legend(["male", 'female'])
    plt.title(name)
    plt.show()

# grade
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)
    for i in range(4):
        sns.kdeplot(data[user["grade"] == i], label=f"{i+1}")
    
    plt.legend()
    plt.title(name)
    plt.show()

# recess
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)
    for i in range(2):
        sns.kdeplot(data[user["recess"] == i])
    
    plt.legend(["no", "yes"])
    plt.title(name)
    plt.show()
    
# t test
import scipy.stats as stats

ttest_results = []

# sex
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)

    t, p = stats.ttest_ind(data[user["sex"] == 0], data[user["sex"] == 1])
    if p < 0.05:
        print(f"sex: male, data: {name}\t |t: {t:.5}, p-value: {p:.2}")
    ttest_results.append(["sex", "male", name, t, p])

# recess
for name, data in zip(data_names, datas):
    data = data.mean(axis=1)

    t, p = stats.ttest_ind(data[user["recess"] == 0], data[user["recess"] == 1])
    if p < 0.05:
        print(f"recess: no, data: {name}\t |t: {t:.5}, p-value: {p:.2}")
    ttest_results.append(["recess", "no", name, t, p])

ttest_df = pd.DataFrame(ttest_results, columns=["group", "H0", "indices", "t", "p-value"])
print(ttest_df)

ttest_df[ttest_df["p-value"] < .05]
ttest_df.to_csv("r_ttest.csv", index=None)

# ANOVA
import scipy.stats as stats

anova_results = []

for name, data in zip(data_names, datas):
    data = data.mean(axis=1)

    f, p = stats.f_oneway(data[user["major"] == 0],
                          data[user["major"] == 1],
                          data[user["major"] == 3],
                          data[user["major"] == 4],
                          data[user["major"] == 5],
                          )
    
    anova_results.append(["major", name, f, p])
    
    print(f"major| data: {name}\t |f: {f:.5}, p-value: {p:.2}")
    
for name, data in zip(data_names, datas):    
    data = data.mean(axis=1)

    # grade
    f, p = stats.f_oneway(data[user["grade"] == 0],
                          data[user["grade"] == 1],
                          data[user["grade"] == 2],
                          data[user["grade"] == 3],
                          )
    anova_results.append(["grade", name, f, p])
    
    print(f"grade| data: {name}\t |f: {f:.5}, p-value: {p:.2}")

anova_df = pd.DataFrame(anova_results, columns=["group", "indices", "F", "p-value"])

print(anova_df)
anova_df[anova_df["p-value"] < .05]

anova_df.to_csv("r_anova.csv", index=None)
