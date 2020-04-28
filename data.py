import numpy as np
import pandas as pd
import shapely.geometry as shg
import geopandas as gpd
import netCDF4 as nc


data = gpd.read_file('./data/landkreise_no2_14d.json')

landkreise = data[data.art == 'landkreis']
bundeslaender = data[data.art == 'bundesland']

rootgroupFebr = nc.Dataset("./data/no2_germany_february2020_1800.nc", "r")
rootgroupMarch = nc.Dataset("./data/no2_germany_march2020_0000.nc", "r")
rootgroupApril = nc.Dataset("./data/no2_germany_april2020_0000.nc", "r")

Tfeb, X, Y = rootgroupFebr['tcno2'].shape
Tmarch, _, _ = rootgroupMarch['tcno2'].shape
Tapril, _, _ = rootgroupApril['tcno2'].shape
Tmeassure = 21
Tdelta = 14

#beforeVals = np.concatenate((rootgroupFebr['tcno2'][-(Tdelta - Tmeassure):], rootgroupMarch['tcno2'][:Tmeassure]))
beforeVals = rootgroupMarch['tcno2'][(Tmeassure - Tdelta):Tmeassure]
afterVals = np.concatenate((rootgroupMarch['tcno2'][Tmeassure:], rootgroupApril['tcno2'][:(Tdelta - (Tmarch - Tmeassure))]))
beforeSum = np.sum(beforeVals, axis=0)
afterSum = np.sum(afterVals, axis=0)

nrLandkreise = len(landkreise.index)
landkreise['obs_no2_before'] = [[] for i in range(nrLandkreise)]
landkreise['obs_no2_after'] = [[] for i in range(nrLandkreise)]

R, C = beforeSum.shape
for r in range(R):
    for c in range(C):
        lat = rootgroupMarch['latitude'][r]
        lon = rootgroupMarch['longitude'][c]
        pt = shg.Point(lon, lat)
        for i in landkreise.index:
            lk = landkreise.loc[i]
            if lk.geometry.contains(pt):
                landkreise.loc[i, 'obs_no2_before'] = landkreise.loc[i, 'obs_no2_before'] +  [beforeSum[r, c]]
                landkreise.loc[i, 'obs_no2_after'] = landkreise.loc[i, 'obs_no2_after'] +  [afterSum[r, c]]
                break


landkreise['obs_no2_midnight_mean_before'] = landkreise['obs_no2_before'].apply(np.mean)
landkreise['obs_no2_midnight_mean_after'] = landkreise['obs_no2_after'].apply(np.mean)


landkreise.drop('obs_no2_before', 1, inplace=True)
landkreise.drop('obs_no2_after', 1, inplace=True)
landkreise.to_file("./data/landkreise_no2_14d.json", driver="GeoJSON", encoding="utf-8")
