# %%
import numpy as np
import pandas as pd
import shapely.geometry as shg
import geopandas as gpd
import netCDF4 as nc

# %%
data = gpd.read_file('./data/landkreise_no2_14d.json')

landkreise = data[data.art == 'landkreis']
bundeslaender = data[data.art == 'bundesland']

dataFebr1800 = nc.Dataset("./data/no2_germany_february2020_1800.nc", "r")
dataFebr0000 = nc.Dataset("./data/no2_germany_february2020_0000.nc", "r")
dataMarch1800 = nc.Dataset("./data/no2_germany_march2020_1800.nc", "r")
dataMarch0000 = nc.Dataset("./data/no2_germany_march2020_0000.nc", "r")
dataApril1800 = nc.Dataset("./data/no2_germany_april2020_1800.nc", "r")
dataApril0000 = nc.Dataset("./data/no2_germany_april2020_0000.nc", "r")

Tfeb, X, Y = dataFebr1800['tcno2'].shape
Tmarch, _, _ = dataMarch1800['tcno2'].shape
Tapril, _, _ = dataApril1800['tcno2'].shape
Tmeassure = 21
Tdelta = 30


# %%
beforeVals1800 = np.concatenate((dataFebr1800['tcno2'][-(Tdelta - Tmeassure):], dataMarch1800['tcno2'][:Tmeassure]))
afterVals1800 = np.concatenate((dataMarch1800['tcno2'][Tmeassure:], dataApril1800['tcno2'][:(Tdelta - (Tmarch - Tmeassure))]))
beforeVals0000 = np.concatenate((dataFebr0000['tcno2'][-(Tdelta - Tmeassure):], dataMarch0000['tcno2'][:Tmeassure]))
afterVals0000 = np.concatenate((dataMarch0000['tcno2'][Tmeassure:], dataApril0000['tcno2'][:(Tdelta - (Tmarch - Tmeassure))]))

beforeSum1800 = np.sum(beforeVals1800, axis=0)
afterSum1800 = np.sum(afterVals1800, axis=0)
beforeSum0000 = np.sum(beforeVals0000, axis=0)
afterSum0000 = np.sum(afterVals0000, axis=0)



# %%
nrLandkreise = len(landkreise.index)
landkreise['no2_1800_before'] = [[] for i in range(nrLandkreise)]
landkreise['no2_1800_after'] = [[] for i in range(nrLandkreise)]
landkreise['no2_0000_before'] = [[] for i in range(nrLandkreise)]
landkreise['no2_0000_after'] = [[] for i in range(nrLandkreise)]

R, C = beforeSum1800.shape
for r in range(R):
    for c in range(C):
        lat = dataMarch1800['latitude'][r]
        lon = dataMarch1800['longitude'][c]
        pt = shg.Point(lon, lat)
        for i in landkreise.index:
            lk = landkreise.loc[i]
            if lk.geometry.contains(pt):
                landkreise.loc[i, 'no2_1800_before'] = landkreise.loc[i, 'no2_1800_before'] +  [beforeSum1800[r, c]]
                landkreise.loc[i, 'no2_1800_after'] = landkreise.loc[i, 'no2_1800_after'] +  [afterSum1800[r, c]]
                landkreise.loc[i, 'no2_0000_before'] = landkreise.loc[i, 'no2_0000_before'] +  [beforeSum0000[r, c]]
                landkreise.loc[i, 'no2_0000_after'] = landkreise.loc[i, 'no2_0000_after'] +  [afterSum0000[r, c]]
                break


# %%
landkreise['obs_no2_mean_before'] = landkreise['no2_1800_before'].apply(np.mean)
landkreise['obs_no2_mean_after'] = landkreise['no2_1800_after'].apply(np.mean)
landkreise['obs_no2_midnight_mean_before'] = landkreise['no2_0000_before'].apply(np.mean)
landkreise['obs_no2_midnight_mean_after'] = landkreise['no2_0000_after'].apply(np.mean)

#%%
landkreise.drop('no2_1800_before', 1, inplace=True)
landkreise.drop('no2_1800_after', 1, inplace=True)
landkreise.drop('no2_0000_before', 1, inplace=True)
landkreise.drop('no2_0000_after', 1, inplace=True)

landkreise.to_file("./data/landkreise_no2_14d.json", driver="GeoJSON", encoding="utf-8")
