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



def calcConnectivity(geometries, u1, u2):
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
                c = u1
            elif neighbors2nd[x, y]:
                c = u2
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



"""
    generic spatial model
"""
def spatialModel(alpha0, alpha1, connectivity0, connectivity1, 
                 nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn):
    
    infected = np.zeros((nrPlaces, nrTimesteps))
    infected[:, 0] = n0
    
    for t in range(nrTimesteps-1):
        alpha = alpha0 if t < T_stepAlpha else alpha1
        connectivity = connectivity0 if t < T_stepConn else connectivity1
        
        n_t = infected[:, t]
        n_w = connectivity.dot(n_t)
        dndt = alpha * n_w * (1 - n_t / Ks)
        infected[:, t+1] = n_t + dndt

    return infected


"""
    connectivity1 is a fraction of connectivity0
"""
def spatialModelFracConn(alpha0, fractionAlpha, fractionSpatial1, fractionSpatial2, fractionConnectivity, 
                        nrPlaces, nrTimesteps, Ks, n0, geometries, T_stepAlpha, T_stepConn):

    alpha1 = alpha0 * fractionAlpha
    connectivity0 = calcConnectivity(geometries, fractionSpatial1, fractionSpatial2)
    connectivity1 = connectivity0 * fractionConnectivity

    return spatialModel(alpha0, alpha1, connectivity0, connectivity1,
                        nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn)


"""
    connectivity1 is estimated from connectivity0 * fraction_NO2
"""
def spatialModelNO2(alpha0, fractionAlpha, fractionSpatial1, fractionSpatial2, fractionNo2Traffic,
                    nrPlaces, nrTimesteps, Ks, n0, geometries, T_stepAlpha, T_stepConn, no2_before, no2_after):
    
    no2_nontraffic = (1 - fractionNo2Traffic) * no2_before
    no2_traffic_before = no2_before - no2_nontraffic
    no2_traffic_after = no2_after - no2_nontraffic
    no2_fraction = no2_traffic_after / no2_traffic_before

    alpha1 = alpha0 * fractionAlpha
    connectivity0 = calcConnectivity(geometries, fractionSpatial1, fractionSpatial2)
    connectivity1 = calcReducedConnectivity(connectivity0, no2_fraction)

    return spatialModel(alpha0, alpha1, connectivity0, connectivity1,
                        nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn)



def estimateSpatialAlphas(values, Ks, connectivity):
    values[values == 0] = 1  # for numeric reasons
    n_t = values[:, :-1]
    n_t1 = values[:, 1:]
    _, nrTimeSteps = n_t.shape
    n_w = connectivity.dot(n_t)
    K_sq = np.repeat([Ks], nrTimeSteps, axis=0).T
    invFracFree = K_sq / (K_sq - n_t)
    alphas_t = invFracFree * (n_t1 - n_t) / n_w
    alphas = np.mean(alphas_t, axis=1)
    return alphas


"""
    connectivity1 is estimated from connectivity0 * fraction_NO2
    alphas are calculated for each LK individually from measurements
"""
def spatialModelNO2alpha(fractionSpatial1, fractionSpatial2, fractionNo2Traffic,
                         Ks, geometries, T_stepAlpha, T_stepConn, infectedMeasured, no2_before, no2_after, fullOutput = False):

    no2_nontraffic = (1 - fractionNo2Traffic) * no2_before
    no2_traffic_before = no2_before - no2_nontraffic
    no2_traffic_after = no2_after - no2_nontraffic
    no2_fraction = no2_traffic_after / no2_traffic_before
    
    nrPlaces, nrTimesteps = infectedMeasured.shape
    n0 = infectedMeasured[:, 0]

    infectedMeasuredBefore = infectedMeasured[:, :T_stepAlpha]
    infectedMeasuredAfter = infectedMeasured[:, T_stepAlpha:]

    connectivity0 = calcConnectivity(geometries, fractionSpatial1, fractionSpatial2)
    connectivity1 = calcReducedConnectivity(connectivity0, no2_fraction)
    
    alphas0 = estimateSpatialAlphas(infectedMeasuredBefore, Ks, connectivity0)
    alphas1 = estimateSpatialAlphas(infectedMeasuredAfter, Ks, connectivity1) 

    sim = spatialModel(alphas0, alphas1, connectivity0, connectivity1,
                        nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn)

    if fullOutput:
        return sim, connectivity0, connectivity1, alphas0, alphas1
    else:
        return sim




def msse(n_obs, n_sim):
    err2 = (n_obs - n_sim)**2
    err2_sum = np.sum(err2, axis=1)
    return np.sum(err2_sum)



def getMsseRelative(norm_by):
    # mse might give extra weight to large cities. This method normalizes all errors by <norm_by>.
    def msseRelative(n_obs, n_sim):
        _, nrTimeSteps = n_obs.shape
        norm_by_sq = np.repeat([norm_by], nrTimeSteps, axis=0).T
        n_obs_norm = n_obs / norm_by_sq
        n_sim_norm = n_sim / norm_by_sq
        return msse(n_obs_norm, n_sim_norm)
    return msseRelative


"""
    @y_obs
    @model
    @startparas: initial guess at the values of the parameters to be calibrated
    @bounds: possible bounds for parameters to be calibrated
    @sparas: static parameters that will *not* be calibrated
    @errorMeassure: function of y_obs and y_sim to be minimized (usually mse)
"""
def minimize(y_obs, model, startparas, bounds, sparas, errorMeassure):
    history = []
    def wrappedObjective(vparas):
        y_sim = model(*vparas, *sparas)
        err = errorMeassure(y_obs, y_sim)
        history.append({'paras': vparas, 'error': err})
        return err
    results = scpo.minimize(wrappedObjective, x0=startparas, bounds=bounds)
    return results, history
