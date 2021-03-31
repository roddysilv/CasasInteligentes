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
from sklearn.cluster import AffinityPropagation
af = AffinityPropagation(preference=None).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = A[cluster_centers_indices[k]]
    plt.plot(A[class_members, 0], A[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
    for x in A[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

for x,s in zip(A,r): 
    pl.text(x=x[0], y=x[1], s=s, fontsize=8)  
    
#pl.axis('equal');
plt.title('Affinity Propagation\nEstimated number of clusters: %d' % n_clusters_)
plt.show()


#%%
from sklearn.cluster import OPTICS, cluster_optics_dbscan
clust = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05)
clust.fit(X)
space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]
n_clusters_ = len(np.unique(clust.labels_[clust.labels_ != -1]))

colors = ['go', 'ro', 'bo', 'yo', 'co']
plt.figure()
for klass, color in zip(range(0, 5), colors):
    Xk = A[clust.labels_ == klass]
    plt.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.99)
   
for x,s in zip(A,r): 
    pl.scatter(x=x[0], y=x[1], s=0,)
    pl.text(x=x[0], y=x[1], s=s, fontsize=8)  
    
plt.plot(A[clust.labels_ == -1, 0], A[clust.labels_ == -1, 1], 'ko',alpha=0.2)
plt.title('OPTICS\nEstimated number of clusters: %d' % n_clusters_)
plt.show()

#%%
houseInfo = pd.read_csv('csv/Houses_info.csv')
houseInfo = pd.concat([houseInfo,
                       pd.get_dummies(houseInfo['HouseType']),
                       pd.get_dummies(houseInfo['Facing']),
                       pd.get_dummies(houseInfo['Region'])],
                       axis=1)



# for c in ['HouseType']:
#     h = pd.get_dummies(houseInfo[c])
#     #houseInfo.drop(columns=c, inplace=True)
#     for c1 in h.columns:
#         houseInfo[c1] = h[c1].values

houseInfo = houseInfo.drop(columns =['HouseType','Facing','Region','FirstReading','LastReading'])

houseInfo = houseInfo.fillna(0)
houseInfo["Cover"] = houseInfo["Cover"].str.replace(",",".").astype(float)
#%%

r2 = houseInfo['House']
D2 = houseInfo.drop(['House'], axis=1)
D2.index = r2

D3 = houseInfo.drop(18, axis=0)
D4 = houseInfo.drop(17, axis=0)

D5 = D2.drop(18, axis=0)
D6 = D5.drop(17, axis=0)
            
for aux in [D2, D3, D4, D5, D6]:
    pca = PCA(n_components=2)
    pca.fit(aux)
    A2 = pca.fit_transform(aux)
    
    pl.figure()
    pl.scatter(A2[:, 0], A2[:, 1], alpha=0.6, s=100)
    for x,s in zip(A2,aux.index): 
        pl.text(x=x[0], y=x[1], s=s, fontsize=10)  
        
    #pl.axis('equal');
    pl.show()
    
#%%