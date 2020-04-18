import numpy as np
import pandas as pd
import shapely.geometry as shg
import geopandas as gpd
import netCDF4 as nc


shps = gpd.read_file('./data/landkreise_risklayer.geojson')
shps = shps.rename(columns={'type': 'art', 'ags': 'AGS'})
shps['AGS'] = shps['AGS'].astype(np.int64)
vals = pd.read_csv('./data/values_landkreise_0904_no_ka.csv')

data = pd.merge(shps, vals, left_on='AGS', right_on='AGS', how='left')
data = data.set_index('name')

dateColNames = data.columns[6:-2]

landkreise = data[data.art == 'landkreis']
landkreise[dateColNames] = landkreise[dateColNames].astype(np.int64)
landkreise.current = landkreise.current.astype(np.int64)

bundeslaender = data[data.art == 'bundesland']
bundeslaender[dateColNames] = landkreise.groupby('partof')[dateColNames].sum()
bundeslaender.current = landkreise.groupby('partof').current.sum()


incubationTime = 7
tCurfew = list(dateColNames.values).index('22.03.2020')
tCurfewEffect = tCurfew + incubationTime


rootgroupFebr = nc.Dataset("./data/no2_germany_february2020_1800.nc", "r")
rootgroupMarch = nc.Dataset("./data/no2_germany_march2020_1800.nc", "r")
rootgroupApril = nc.Dataset("./data/no2_germany_april2020_1800.nc", "r")

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


landkreise['obs_no2_mean_before'] = landkreise['obs_no2_before'].apply(np.mean)
landkreise['obs_no2_mean_after'] = landkreise['obs_no2_after'].apply(np.mean)


fractionNo2Traffic = 0.5
landkreise['NO2_non_traffic'] = landkreise['obs_no2_mean_before'] * (1 - fractionNo2Traffic)
landkreise['NO2_traffic_before'] = landkreise['obs_no2_mean_before'] * fractionNo2Traffic
landkreise['NO2_traffic_after'] = landkreise['obs_no2_mean_after'] - landkreise['NO2_non_traffic']
landkreise['NO2_diff'] = landkreise['NO2_traffic_after'] - landkreise['NO2_traffic_before']
landkreise['NO2_diff_frac'] = landkreise['NO2_traffic_after'] / landkreise['NO2_traffic_before']

landkreise.drop('obs_no2_before', 1, inplace=True)
landkreise.drop('obs_no2_after', 1, inplace=True)
landkreise.to_file("./data/landkreise_no2_14d.json", driver="GeoJSON", encoding="utf-8")
