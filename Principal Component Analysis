import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# data load
X = pd.read_csv("data_fa.csv")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
n = 8

# Cumulative Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1),
         pca.explained_variance_ratio_[:],
         marker='o',
         linestyle='-',
         color='k')
plt.vlines(n, 0, pca.explained_variance_ratio_[0], linestyle='--', color='r')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.savefig("scree_plot.png")
plt.grid(True)
plt.show()

pca_X = PCA(n_components=n)
X_pca = pd.DataFrame(pca_X.fit_transform(X_scaled))
explained_variance_ratio = pca_X.explained_variance_ratio_.cumsum()
print(explained_variance_ratio)

pc_loadings = pca_X.components_

for j in range(len(pc_loadings)):
    print("PC", j+1)
    for i, loading in enumerate(pc_loadings[j]):
        print(rf"{loading:.3f} \times factor_{i}", end=' + ')
    print("="*54)

print(pc_loadings)
df_pc_loadings = pd.DataFrame(pc_loadings)
df_pc_loadings.to_csv("pc_loadings.csv", index=None)
X_pca.to_csv("data_pca.csv", index=None)
