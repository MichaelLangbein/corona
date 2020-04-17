import numpy as np
import pandas as pd
import geopandas as gpd



shps = gpd.read_file('./data/landkreise_risklayer.geojson')
shps = shps.rename(columns={'type': 'art', 'ags': 'AGS'})
shps['AGS'] = shps['AGS'].astype(np.int64)
vals = pd.read_csv('./data/values_landkreise_0904_no_ka.csv')

data = pd.merge(shps, vals, left_on='AGS', right_on='AGS', how='left')
data = data.set_index('name')

dateColNames = data.columns[6:-2]

bundeslaender = data[data.art == 'bundesland']
landkreise = data[data.art == 'landkreis']
landkreise[dateColNames] = landkreise[dateColNames].astype(np.int64)
bundeslaender[dateColNames] = landkreise.groupby('partof')[dateColNames].sum()
landkreise.current = landkreise.current.astype(np.int64)
bundeslaender.current = landkreise.groupby('partof').current.sum()

laenderNames = bundeslaender.index
laenderIds = np.arange(len(laenderNames))
bayernCuml = bundeslaender.loc['Bayern']
KBayern = bayernCuml['population']


landkreiseBayern = landkreise[landkreise.partof == 'Bayern']
landkreisNamesBayern = landkreiseBayern.index
landkreisIdsBayern = np.arange(len(landkreisNamesBayern))
populationBayern = landkreiseBayern['population'].values
KsBayern = landkreiseBayern['population']
T = len(dateColNames)
L = len(landkreisIdsBayern)
time = np.arange(T)
incubationTime = 7
tCurfew = list(dateColNames.values).index('22.03.2020')
tCurfewEffect = tCurfew + incubationTime

lksNo2 = gpd.read_file("./data/landkreise_no2_14d.json", driver="GeoJSON")
lksNo2 = lksNo2.set_index('name')