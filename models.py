import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.optimize as scpo
import shapely as sply
from utils import tabling, logging




def simpleModel(alpha, 
                nrPlaces, nrTimesteps, Ks, n0):
    infected = np.zeros((nrPlaces, nrTimesteps))
    infected[:, 0] = n0
    for t in range(nrTimesteps-1):
        dndt = alpha * infected[:, t] * (1 - infected[:, t] / Ks)
        infected[:, t+1] = infected[:, t] + dndt
    return infected


def stepModel(alpha0, fractionAlpha, 
              nrPlaces, nrTimesteps, Ks, n0, T_stepAlpha):
    infected = np.zeros((nrPlaces, nrTimesteps))
    infected[:, 0] = n0
    alpha1 = alpha0 * fractionAlpha
    for t in range(nrTimesteps-1):
        alpha = alpha0 if t < T_stepAlpha else alpha1
        dndt = alpha * infected[:, t] * (1 - infected[:, t] / Ks)
        infected[:, t+1] = infected[:, t] + dndt
    return infected



def getNeighbors0(L):
    return np.eye(L, dtype=bool)


def getNeighbors1(geometries):
    L = len(geometries)
    neighbors1st = np.zeros((L, L), dtype=bool)
    for x in range(L):
        for y in range(x, L):
            if geometries[x].touches(geometries[y]):
                neighbors1st[x, y] = True
                neighbors1st[y, x] = True
    return neighbors1st


@tabling
def getNeighborsNth(geometries, N):
    L = len(geometries)

    if N == 0:
        return getNeighbors0(L)
    if N == 1:
        neighbors0 = getNeighbors0(L)
        neighbors1 = getNeighbors1(geometries)
        return neighbors1 * ~neighbors0

    neighborsNmin1 = getNeighborsNth(geometries, N-1)
    neighborsN = neighborsNmin1.dot(neighborsNmin1) * ~neighborsNmin1
    return neighborsN


def calcConnectivity(geometries, u):
    L = len(geometries)
    neighbors0th = getNeighborsNth(geometries, 0)
    neighbors1st = getNeighborsNth(geometries, 1)
    neighbors2nd = getNeighborsNth(geometries, 2)
    connectivity = np.zeros((L, L), dtype=np.float64)
    for x in range(L):
        for y in range(x, L):
            if neighbors0th[x, y]:
                c = 1
            elif neighbors1st[x, y]:
                c = u
            elif neighbors2nd[x, y]:
                c = u / 2
            else:
                c = 0.0
            connectivity[x, y] = c
            connectivity[y, x] = c
    return connectivity


def calcReducedConnectivity(connectivity, fractionNo2):
    connectivityReduced = np.copy(connectivity)
    X, Y = connectivity.shape
    for x in range(X):
        for y in range(x, Y):
            c = connectivity[x, y]
            if connectivity[x, y] > 0:
                reduction1 = fractionNo2[x]
                reduction2 = fractionNo2[y]
                if not np.isnan(reduction1) and not np.isnan(reduction2):
                    c = c * (reduction1 + reduction2) / 2.0
            connectivityReduced[x, y] = c
            connectivityReduced[y, x] = c
    return connectivityReduced


def spatialModel(alpha0, fractionAlpha, fractionSpatial, fractionConnectivity, 
                 nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn):
    infected = np.zeros((nrPlaces, nrTimesteps))
    infected[:, 0] = n0
    alpha1 = alpha0 * fractionAlpha
    connectivity0 = calcConnectivity(geometries, fractionSpatial)
    connectivity1 = connectivity0 * fractionConnectivity
    for t in range(nrTimesteps-1):
        alpha = alpha0 if t < T_stepAlpha else alpha1
        conct = connectivity0 if t < T_stepConn else connectivity1
        for x in range(nrPlaces):
            K = Ks[x]
            nc = np.inner( conct[x, :], infected[:, t] )
            dndt = alpha * nc * (1 - infected[x, t] / K)
            infected[x, t+1] = infected[x, t] + dndt
    return infected


def spatialModelNO2(alpha0, fractionAlpha, fractionSpatial,
                    nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn, fractionNo2):
    infected = np.zeros((nrPlaces, nrTimesteps))
    infected[:, 0] = n0
    alpha1 = alpha0 * fractionAlpha
    connectivity0 = calcConnectivity(geometries, fractionSpatial)
    connectivity1 = calcReducedConnectivity(connectivity0, fractionNo2)
    for t in range(nrTimesteps-1):
        alpha = alpha0 if t < T_stepAlpha else alpha1
        conct = connectivity0 if t < T_stepConn else connectivity1
        for x in range(nrPlaces):
            K = Ks[x]
            nc = np.inner( conct[x, :], infected[:, t] )
            dndt = alpha * nc * (1 - infected[x, t] / K)
            infected[x, t+1] = infected[x, t] + dndt
    return infected


def msse(n_obs, n_sim):
    err2 = (n_obs - n_sim)**2
    err2_sum = np.sum(err2, axis=1)
    return np.mean(err2_sum)


def msseRelative(n_obs, n_sim, norm_by):
    # mse might give extra weight to large cities. This method normalizes all errors by population.
    err2 = (n_obs - n_sim)**2
    err2_sum = np.sum(err2, axis=1)
    err2_norm = err2_sum / norm_by
    return np.mean(err2_norm)


def minimize(y_obs, model, vparas, sparas, startparas, bounds, errorMeassure):
    history = []
    def wrappedObjective(vparas):
        y_sim = model(*vparas, *sparas)
        err = errorMeassure(y_obs, y_sim)
        history.append({'paras': vparas, 'error': err})
        return err
    results = scpo.minimize(wrappedObjective, x0=startparas, bounds=bounds)
    results.history = history
    return results
