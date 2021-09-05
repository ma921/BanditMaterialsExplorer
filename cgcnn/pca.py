from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

df = pd.read_csv('./results/cgcnn_d128_MAE134.csv', index_col=0)
X = np.array(df)
pca = PCA(n_components=12).fit(X)
results = pca.transform(X)
df_conv = pd.DataFrame(results, index=df.index)
df_conv.to_csv('descriptors.csv')

