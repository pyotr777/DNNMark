#!/usr/bin/env python3

# Library of plotting functions

# 2018-2023 (C) Peter Bryzgalov @ CHITECH Stair Lab

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
import math

print("Plotter library v.1.03e")


def plotHeatMap(df, title=None, cmap=None, ax=None, zrange=None, format=".3f"):
    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        cmap = "viridis"
    if zrange is None:
        cmesh = ax.pcolormesh(df, cmap=cmap)
    else:
        cmesh = ax.pcolormesh(df, cmap=cmap, vmin=zrange[0], vmax=zrange[1])

    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns)
    ax.tick_params(direction='in', length=0, pad=10)
    for y in range(df.shape[0]):
        for x in range(df.shape[1]):
            # if df.iloc[y,x]  0:
            ax.text(x + 0.5, y + 0.5, '{0:{fmt}}'.format(df.iloc[y, x], fmt=format), color="black",
                    fontsize=9, horizontalalignment='center', verticalalignment='center', bbox={
                        'facecolor': 'white',
                        'edgecolor': 'none',
                        'alpha': 0.2,
                        'pad': 0
                    })
    ax.set_title(title, fontsize=16)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    cbar = plt.colorbar(cmesh, ax=ax, pad=0.02)
    cbar.ax.tick_params(direction='out', length=3, pad=5)
    return (ax, cbar)


def getColorList(cmap, n):
    cmap = cm.get_cmap(cmap, n)
    colors = []
    for i in range(cmap.N):
        c = matplotlib.colors.to_hex(cmap(i), keep_alpha=True)
        colors.append(c)
    return colors


def rotateXticks(ax, angle):
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)


def rotateYticks(ax, angle):
    for tick in ax.get_yticklabels():
        tick.set_rotation(angle)


def testColorMap(cmap):
    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 1.5 * np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt((X + Y) * (X + Y + 2) * 0.3)

    fig, ax = plt.subplots(figsize=(2, 1.6))
    im = ax.imshow(Z, aspect="auto", interpolation='nearest', vmin=0, origin='lower', cmap=cmap)
    fig.colorbar(im, ax=ax)
    # plt.show()


# Plot grid on the axis ax
def drawGrid(ax, xstep=None, ystep=None, minor_ticks_x=2, minor_ticks_y=2):
    # ax.set_xlim(0, None)
    # ax.set_ylim(0, None)
    ax.minorticks_on()
    if xstep is not None:
        minorLocatorX = MultipleLocator(xstep / minor_ticks_x)
        majorLocatorX = MultipleLocator(xstep)
        ax.xaxis.set_major_locator(majorLocatorX)
        ax.xaxis.set_minor_locator(minorLocatorX)
    if ystep is not None:
        minorLocatorY = MultipleLocator(ystep / minor_ticks_y)
        majorLocatorY = MultipleLocator(ystep)
        ax.yaxis.set_minor_locator(minorLocatorY)
        ax.yaxis.set_major_locator(majorLocatorY)
    ax.grid(which='major', ls=":", lw=.5, alpha=0.8, color="grey")
    ax.grid(which='minor', ls=':', lw=.2, alpha=.8, color='grey')


# Colors
prediction_plot_colors = {
    "targetCNN": "tab:blue",
    "proxyApp": "#92cde3",
    "delta": "#7bba45",
    "predictions": "tab:red",
    "nonGPU": "#b47c98",
    "GPU": "#549f76"
}


# Plot prediction and errors
# DF must have columns:
# 'batch', 'time', 'machine',
# 'ver' - framework name and version
# evaluator,
# 'CPU time predicted', 'GPU time predicted', 'H',
# 'AE', 'APE'
# save - filename to save the plot
def plotPredictionsAndErrors(df_, train_set, MAE, MAPE, evaluator, AE="AE", APE="APE", ver=None,
                             model='VGG16', mbs_range=None, save=None, title=None, CPUpredictions=False,
                             GPUpredictions=False, text=None):
    colors = prediction_plot_colors
    # Test if df includes necessary columns
    nscolumns = ['batch', 'machine', evaluator, evaluator + ' predicted', AE, APE]
    if CPUpredictions:
        nscolumns.append("CPU time predicted")
    if GPUpredictions:
        nscolumns.append("GPU time predicted")
    columns = df_.columns
    for col in nscolumns:
        if col not in columns:
            print("ERROR: Dataframe df must include column {}".format(col))
            return

    df = df_.copy()
    machine = df.iloc[0]['machine']
    # print("plotPredictionsAndErrors. df=")
    # print(df.head())
    fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True, dpi=144)
    ax = axs[0]

    # AE in milliseconds
    AEms = AE + "ms"
    df.loc[:, AEms] = df[AE] * 1000.

    if mbs_range is not None:
        left, right = mbs_range
        df = df[df["batch"] <= right]
        df = df[df["batch"] >= left]
        train_set = train_set[train_set["batch"] <= right]
        train_set = train_set[train_set["batch"] >= left]

    if CPUpredictions:
        if "CPU time predicted" in df.columns:
            df.plot("batch", "CPU time predicted", c=colors["nonGPU"], lw=1, ls=":",
                    label="non-GPU time predictions", ax=ax)

    if GPUpredictions:
        df.plot("batch", "GPU time predicted", c=colors["GPU"], lw=1, ls=":",
                label="GPU time predictions", ax=ax)

    train_set.plot("batch", evaluator, lw=0, marker="o", ms=6, color=colors["targetCNN"], mfc='white',
                   mec='black', mew=1, label="Used samples", ax=ax)

    kwargs = {}
    if mbs_range is not None:
        kwargs = {'marker': "o", 'ms': 5, 'mfc': "white"}
    df.plot("batch", evaluator, label="Target CNN time", c=colors["targetCNN"], lw=0.8, ax=ax, **kwargs)

    df.plot("batch", evaluator + ' predicted', c=colors["predictions"], lw=0.8, label="predictions",
            ax=ax, **kwargs)

    ax.set_ylabel("time (s)")
    ax.set_xlabel("")
    if title is None:
        ax.set_title("{} {} {} predictions and errors on {}".format(ver.capitalize(), model, machine,
                                                                    evaluator))
    else:
        ax.set_title(title)
    ax.legend(fontsize=10, frameon=False, bbox_to_anchor=(1, 1))
    if mbs_range is not None:
        if right - left < 100:
            drawGrid(ax, xstep=10, minor_ticks_x=10)
    else:
        drawGrid(ax, minor_ticks_y=5)

    ax = axs[1]
    ax.set_ylabel("ms")
    ax.set_ylim(0, 35)
    ax1 = ax.twinx()
    df.plot.area("batch", APE, color="coral", lw=1, alpha=0.6, label="APE (%)", ax=ax1)

    df.plot("batch", AEms, lw=1, ls="-", color="black", label="AE (ms)", ax=ax)
    maxAPE = df[APE].max()

    ax1.set_ylabel("%")
    ax1.set_ylim(0, 35)
    if mbs_range is not None:
        if right - left < 100:
            drawGrid(ax, xstep=10, minor_ticks_x=10, ystep=10, minor_ticks_y=5)
    else:
        drawGrid(ax, ystep=10, minor_ticks_y=5)

    ax.legend(fontsize=10, loc='upper right', frameon=False, bbox_to_anchor=(1., 1))
    ax1.legend(fontsize=10, loc='upper right', frameon=False, bbox_to_anchor=(1., .85))
    # MAPE and MAE
    ax1.text(1.06, 1, "MAPE {:.2f}%\nMAE  {:.2f}ms\nmax APE {:.2f}%".format(MAPE, MAE * 1000., maxAPE),
             transform=ax1.transAxes, va='top')
    if text is not None:
        ax.text(0, -0.35, text, va="top", ha="left", transform=ax.transAxes)
    ax.set_xlabel("mini-batch size")
    ax.set_xlim(0, None)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, bbox_inches="tight")
        print("Saved plot to", save)

    plt.close()
    # return fig, ax


def plotColortable(colors, ncols=4, sort_colors=True):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(colors,
                       key=lambda c: tuple(matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height, (width - margin) / width,
                        (height - margin) / height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14, horizontalalignment='left', verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width, height=18, facecolor=colors[i],
                      edgecolor='0.7'))

    return fig
