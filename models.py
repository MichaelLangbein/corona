import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.optimize as scpo
import scipy.linalg as scpl
import shapely as sply
from utils import tabling, logging



#   A * A_ri = I_r
# rxc   cxr
# r <= c

def rightInverse(x):
    R, C = x.shape
    if R > C:
        raise Exception('Right inverse only exists for broad matrices')
    xT = x.T
    x_xT = np.matmul(x, xT)
    x_xT_inv = np.linalg.inv(x_xT)
    ri = np.matmul(xT, x_xT_inv)
    return ri


def leftInverse(x):
    R, C = x.shape
    if R < C:
        raise Exception('Left inverse only exists for tall matrices')
    xT = x.T
    xT_x = np.matmul(xT, x)
    xT_x_inv = np.linalg.inv(xT_x)
    li = np.matmul(xT_x_inv, xT)
    return li


def calcChangeNormd(n_obs, Ks, delta=1):
    nrPlaces, nrTimesteps = n_obs.shape
    change_normd = np.zeros((nrPlaces, nrTimesteps))
    for x in range(nrPlaces):
        for t in range(nrTimesteps):
            nt = n_obs[x, t]
            ntm1 = n_obs[x, t - delta] if (t - delta)>=0 else 0
            change_normd[x, t] = (Ks[x] / (Ks[x] - nt)) * (nt - ntm1)
    return change_normd


def estimateConnectivityFromObs(n_obs, Ks):
    change_normd = calcChangeNormd(n_obs, Ks)
    n_ri = rightInverse(n_obs)
    conn = change_normd @ n_ri
    return conn


def estimateAlphaFromConAndObs(n_obs, Ks, conn):
    L, _ = n_obs.shape
    change_normd = calcChangeNormd(n_obs, Ks)
    CN = conn @ n_obs
    alphas = change_normd / CN
    alpha = np.zeros(L)
    for l in range(L):
        mean = np.mean([a for a in alphas[l, :] if np.isfinite(a)])
        if np.isfinite(mean):
            alpha[l] = mean
        else:
            alpha[l] = 0
    return alpha


def splitOutAlpha(conn):
    nrPlaces, _ = conn.shape
    alphas = np.diag(conn)
    conn1 = np.zeros((nrPlaces, nrPlaces))
    for r in range(nrPlaces):
        for c in range(nrPlaces):
            conn1[r, c] = conn[r, c] / alphas[r]
    return alphas, conn1



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



@tabling
def getNeighbors1(geometries):
    L = len(geometries)
    neighbors1st = np.zeros((L, L), dtype=bool)
    for x in range(L):
        for y in range(x, L):
            if geometries[x].touches(geometries[y]):
                neighbors1st[x, y] = True
                neighbors1st[y, x] = True
    return neighbors1st



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
    neighbors0th = getNeighborsNth(geometries, 0)
    neighbors1st = getNeighborsNth(geometries, 1)
    neighbors2nd = getNeighborsNth(geometries, 2)
    connectivity = 1 * neighbors0th + u1 * neighbors1st + u2 * neighbors2nd
    return connectivity



def calcReducedConnectivity(connectivity, fractionNo2):
    r = np.sqrt(np.outer(fractionNo2, fractionNo2))
    connectivityReduced = r * connectivity
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
def spatialModelNO2(alpha0, fractionAlpha, fractionSpatial1, fractionSpatial2, fractionTrafficMidnight, 
                    nrPlaces, nrTimesteps, Ks, n0, geometries, T_stepAlpha, T_stepConn, 
                    no2_noon_before, no2_noon_after, no2_night_before, no2_night_after):

    z = np.zeros(nrPlaces)
    no2_base = (1 - fractionTrafficMidnight) * no2_night_after
    no2_traffic_before = np.max((no2_noon_before - no2_base, z), axis=0)
    no2_traffic_after = np.max((no2_noon_after - no2_base, z), axis=0)
    no2_fraction = no2_traffic_after / no2_traffic_before

    alpha1 = alpha0 * fractionAlpha
    connectivity0 = calcConnectivity(geometries, fractionSpatial1, fractionSpatial2)
    connectivity1 = calcReducedConnectivity(connectivity0, no2_fraction)

    return spatialModel(alpha0, alpha1, connectivity0, connectivity1,
                        nrPlaces, nrTimesteps, n0, Ks, geometries, T_stepAlpha, T_stepConn)




"""
    connectivity1 is estimated from connectivity0 * fraction_NO2
    alphas are calculated for each LK individually from measurements
"""
def spatialModelNO2alpha(fractionSpatial1, fractionSpatial2, fractionTrafficMidnight,
                         Ks, geometries, T_stepAlpha, T_stepConn, infectedMeasured,
                         no2_noon_before, no2_noon_after, no2_night_before, no2_night_after,
                         fullOutput = False):

    nrPlaces, nrTimesteps = infectedMeasured.shape

    z = np.zeros(nrPlaces)
    no2_base = (1 - fractionTrafficMidnight) * no2_night_after
    no2_traffic_before = np.max((no2_noon_before - no2_base, z), axis=0)
    no2_traffic_after = np.max((no2_noon_after - no2_base, z), axis=0)
    no2_fraction = no2_traffic_after / no2_traffic_before
    
    n0 = infectedMeasured[:, 0]
    infectedMeasuredBefore = infectedMeasured[:, :T_stepAlpha]
    infectedMeasuredAfter = infectedMeasured[:, T_stepAlpha:]

    connectivity0 = calcConnectivity(geometries, fractionSpatial1, fractionSpatial2)
    connectivity1 = calcReducedConnectivity(connectivity0, no2_fraction)
    
    alphas0 = estimateAlphaFromConAndObs(infectedMeasuredBefore, Ks, connectivity0)
    alphas1 = estimateAlphaFromConAndObs(infectedMeasuredAfter, Ks, connectivity1)

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
