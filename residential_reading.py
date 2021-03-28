import pandas as pd
import numpy as np
import tsfel
from glob import glob

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


