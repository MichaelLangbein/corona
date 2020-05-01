import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable



def getIndex(name, names):
    for i, n in enumerate(names):
        if n == name:
            return i


def plotSim(time, y_obs, y_sim):
    timeV = np.arange(len(time))
    plt.scatter(timeV, y_obs)
    plt.plot(timeV, y_sim)
    plt.xticks(timeV, time, rotation=45)
    for i, label in enumerate(plt.gca().get_xaxis().get_ticklabels()):
        if i%4 != 0:
            label.set_visible(False)
    plt.xlabel('time')
    plt.ylabel('n')


def plotSimMulti(time, y_obs, y_sim, names):
    timeV = np.arange(len(time))
    for i, name in enumerate(names):
        plt.scatter(timeV, y_obs[i])
        plt.plot(timeV, y_sim[i], label=name)
    if len(names) < 6:
        plt.legend()
    plt.xticks(timeV, time, rotation=45)
    for i, label in enumerate(plt.gca().get_xaxis().get_ticklabels()):
        if i%4 != 0:
            label.set_visible(False)
    plt.xlabel('time')
    plt.ylabel('n')


def plotSimCuml(time, y_obs, y_sim):
    y_obs_cuml = np.sum(y_obs, axis=0)
    y_sim_cuml = np.sum(y_sim, axis=0)
    plotSim(time, y_obs_cuml, y_sim_cuml)


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


def plotLocalDifferences(y_obs, y_sim):
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(y_obs)
    axes[0].set_title('data')
    axes[1].imshow((y_obs - y_sim)**2)
    axes[1].set_title('SE')
    axes[2].imshow(y_sim)
    axes[2].set_title('simulation')



def video(title, labels, data):
    T, X, Y = data.shape
    minval = np.min(data)
    maxval = np.max(data)

    fig = plt.figure()
    fig.suptitle(title)
    ax = plt.axes(xlim=(0, Y), ylim=(0, X))
    img = plt.imshow(data[0], animated=True, vmin=minval, vmax=maxval)
    fig.colorbar(img)

    def update(i):
        ax.set_title(labels[i])
        img.set_data(data[i])
        return [img]
    
    ani = FuncAnimation(fig, update, range(T), blit=True)
    return ani


def multiVideo(title, labels, datas):
    nrImages = len(datas)
    T, R, C = datas[0].shape

    fig, axes = plt.subplots(1, nrImages, figsize=(nrImages *4, 4))
    fig.suptitle(title)
    imgs = []
    for i in range(nrImages):
        axes[i].xlim = (0, C)
        axes[i].ylim = (0, R)
        minval = np.min(datas[i])
        maxval = np.max(datas[i])
        img = axes[i].imshow(datas[i][0], animated=True, vmin=minval, vmax=maxval)
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)
        imgs.append(img)

    def update(t):
        for i in range(nrImages):
            axes[i].set_title(labels[i][t])
            imgs[i].set_data(datas[i][t])

    ani = FuncAnimation(fig, update, range(T))
    return ani


def plot2PlusFracNP(title, heading0, heading1, heading2, data0, data1):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    fig.suptitle(title)
    R, C = data0.shape
    allData = np.concatenate((data0, data1))
    minval = np.min(allData)
    maxval = np.max(allData)
    data2 = data1 / data0

    axes[0].xlim = (0, C)
    axes[0].ylim = (0, R)
    axes[0].set_title(heading0)
    img0 = axes[0].imshow(data0, vmin=minval, vmax=maxval)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img0, cax=cax0)

    axes[1].xlim = (0, C)
    axes[1].ylim = (0, R)
    axes[1].set_title(heading1)
    img1 = axes[1].imshow(data1, vmin=minval, vmax=maxval)
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img1, cax=cax1)

    axes[2].xlim = (0, C)
    axes[2].ylim = (0, R)
    axes[2].set_title(heading2)
    img2 = axes[2].imshow(data2)
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img2, cax=cax2)

    return fig, axes


def makeVideo(makeImgFunc, ts):
    fig, axes = makeImgFunc(0)

    def update(t):
        makeImgFunc(t)

    ani = FuncAnimation(fig, update, ts)
    return ani



def plot2PlusFracGeopandas(data, col0, col1):
    minval = np.min(data[col0])
    maxval = np.max(data[col1])
    data[f"frac_{col1}_{col0}"] = data[col1] / data[col0]

    fig, axes = plt.subplots(1, 3, figsize=(20,9))
    divider0 = make_axes_locatable(axes[0])
    divider1 = make_axes_locatable(axes[1])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    data.plot(column=col0, axes=axes[0], legend=True, cax=cax0, vmin=minval, vmax=maxval)
    data.plot(column=col1, axes=axes[1], legend=True, cax=cax1, vmin=minval, vmax=maxval)
    data.plot(column=f"frac_{col1}_{col0}", axes=axes[2], legend=True, scheme='quantiles')
    axes[0].set_title(col0)
    axes[1].set_title(col1)
    axes[2].set_title(f"frac_{col1}_{col0}")

    data.drop(f"frac_{col1}_{col0}", axis=1, inplace=True)