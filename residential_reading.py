import pandas as pd
import numpy as np
import tsfel
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA

#%%
#Don't forget to change this PATH
path = 'csv/'
dataframeempty = pd.DataFrame()
for csv_path in glob(path+'Residential_*.csv'):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df.drop(['date'],axis=1, inplace=True)
    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    # Extract features
    X = tsfel.time_series_features_extractor(cfg, df)
    X['File Name'] = csv_path
    dataframeempty = dataframeempty.append(X)

dataframeempty.to_csv('csv/tsfel_results.csv')

#%%
X = dataframeempty.copy()
X['File Name'] = ['H-'+f.split('/')[-1].split('.csv')[0].split('_')[-1] for f in X['File Name']]

r = X['File Name']
D = X.drop(['File Name'], axis=1)
#%%
pca = PCA(n_components=2)
pca.fit(D)
A = pca.fit_transform(D)
#%%
# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops=dict(arrowstyle='->',
#                     linewidth=2,
#                     shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)


# plt.scatter(A[:, 0], A[:, 1], alpha=0.9)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal');
#%%
plt.figure()
plt.scatter(A[:, 0], A[:, 1], alpha=0.9)
for x,s in zip(A,r): 
    plt.text(x=x[0], y=x[1], s=s, fontsize=10)  
    
plt.axis('equal');

#%%
