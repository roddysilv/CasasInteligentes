import pandas as pd
import numpy as np
import tsfel
from glob import glob
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns; sns.set()

from sklearn.decomposition import PCA

pl.rc('text', usetex=True)
pl.rc('font', family='serif',  serif='Times')    

#%%
#Don't forget to change this PATH
path = 'csv/'
dataframeempty = pd.DataFrame()
W =[]
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


    cfgx = tsfel.get_features_by_domain(domain="spectral")
    cfgs = tsfel.get_features_by_domain(domain="statistical")
    cfgt = tsfel.get_features_by_domain(domain="temporal")
    
    x  = tsfel.time_series_features_extractor(cfg , df, verbose=0)
    xx = tsfel.time_series_features_extractor(cfgx, df, verbose=0)
    xs = tsfel.time_series_features_extractor(cfgs, df, verbose=0)
    xt = tsfel.time_series_features_extractor(cfgt, df, verbose=0)
    
    x=pd.concat([xs,xt,], axis=1)
    W.append(list(x.values.ravel()))


dataframeempty.to_csv('csv/tsfel_results.csv')

W=pd.DataFrame(W, columns=x.columns)

W.to_csv('csv/tsfel_results_uncorrelated.csv')

#%%
X = dataframeempty.copy()

W['File Name']  = X['File Name'].values
#X = W.copy()


cf=tsfel.correlated_features(X)
X.drop(labels=cf, axis=1, inplace=True)
X.drop(labels='0_ECDF Slope', axis=1, inplace=True)


X['File Name'] = [f.split('/')[-1].split('.csv')[0].split('_')[-1] for f in X['File Name']]

r = X['File Name']
D = X.drop(['File Name'], axis=1)
#%%
pca = PCA(n_components=2)
pca.fit(D)
A = pca.fit_transform(D)
#%%
# def draw_vector(v0, v1, ax=None):
#     ax = ax or pl.gca()
#     arrowprops=dict(arrowstyle='->',
#                     linewidth=2,
#                     shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)


# pl.scatter(A[:, 0], A[:, 1], alpha=0.9)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# pl.axis('equal');
#%%
pl.figure()
pl.scatter(A[:, 0], A[:, 1], alpha=0.6, s=1)
for x,s in zip(A,r): 
    pl.text(x=x[0], y=x[1], s=s, fontsize=8)  
    
pl.axis('equal');

#%%
