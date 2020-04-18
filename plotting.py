import numpy as np
import matplotlib.pyplot as plt


def plotSim(time, y_obs, y_sim):
    timeV = np.arange(len(time))
    plt.scatter(timeV, y_obs)
    plt.plot(timeV, y_sim)
    plt.xticks(timeV, time, rotation=90)


def plotSimLks(time, y_obs, y_sim, indices, names):
    timeV = np.arange(len(time))
    for i, name in zip(indices, names):
        plt.scatter(timeV, y_obs[i])
        plt.plot(timeV, y_sim[i], label=name)
    plt.legend()
    plt.xticks(timeV, time, rotation=90)


def plotSimLksCuml(time, y_obs, y_sim):
    timeV = np.arange(len(time))
    y_obs_cuml = np.sum(y_obs, axis=0)
    y_sim_cuml = np.sum(y_sim, axis=0)
    plt.scatter(timeV, y_obs_cuml)
    plt.plot(timeV, y_sim_cuml)
    plt.xticks(timeV, time, rotation=90)


def scatterHistory(hist, p1, p2, filterFunc, l1, l2, cmapName='viridis'):
    xs = []
    ys = []
    zs = []
    for e in hist:
        if filterFunc(e):
            xs.append(e['args'][0][p1])
            ys.append(e['args'][0][p2])
            zs.append(e['val'])
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    img = ax.scatter(xs, ys, c=zs, cmap=plt.get_cmap(cmapName))
    ax.set_xlabel(l1)
    ax.set_ylabel(l2)
    legend1 = ax.legend(*img.legend_elements(),
                    loc="lower left")
    ax.add_artist(legend1)


def plotNo2(lksNo2):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmax = lksNo2['beforeNormal'].max()
    axes[0].vmax = vmax
    axes[1].vmax = vmax
    axes[2].vmax = vmax
    axes[3].vmax = vmax
    axes[0].set_title('before')
    axes[1].set_title('after')
    axes[2].set_title('diff')
    axes[3].set_title('diffFrac')
    lksNo2.plot(column='NO2_traffic_before', axes=axes[0])
    lksNo2.plot(column='NO2_traffic_after', axes=axes[1])
    lksNo2.plot(column='NO2_diff', axes=axes[2])
    lksNo2.plot(column='NO2_diff_frac', axes=axes[3])


def plotLocalDifferences(model, paras):
    y = landkreiseBayern[dateColNames].values
    y_sim = model(*paras)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(y)
    axes[0].set_title('data')
    axes[1].imshow((y - y_sim)**2)
    axes[1].set_title('SE')
    axes[2].imshow(y_sim)
    axes[2].set_title('simulation')

def plotLocalDifferencesRelative(model, paras):
    y = landkreiseBayern[dateColNames].values
    y_sim = model(*paras)
    populationBayernRs = populationBayern.reshape(len(populationBayern), 1)
    y_p = (y / populationBayernRs) * 10000
    y_sim_p = (y_sim / populationBayernRs) * 10000
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(y_p)
    axes[0].set_title('data')
    axes[1].imshow((y_p - y_sim_p)**2)
    axes[1].set_title('SE')
    axes[2].imshow(y_sim_p)
    axes[2].set_title('simulation')


def differenceMap(model, paras):
    y = landkreiseBayern[dateColNames].values
    y_sim = model(*paras)
    lksBdiff = landkreiseBayern.copy()
    lksBdiff['diffSim'] = np.sum(y - y_sim, axis=1)
    lksBdiff.plot(column='diffSim')


def differenceMapRelative(model, paras):
    y = landkreiseBayern[dateColNames].values
    y_sim = model(*paras)
    lksBdiff = landkreiseBayern.copy()
    lksBdiff['diffSim'] = np.sum(y - y_sim, axis=1)
    lksBdiff['diffSimRel'] = lksBdiff['diffSim'] / lksBdiff['population']
    lksBdiff.plot(column='diffSimRel')

def video(data):
    T, X, Y = data.shape

    fig = plt.figure()
    ax = plt.axes(xlim=(0, Y), ylim=(0, X))
    img = plt.imshow(data[0], animated=True)

    def update(i):
        img.set_data(data[i])
        return [img]
    
    ani = FuncAnimation(fig, update, range(T), blit=True)
    return ani