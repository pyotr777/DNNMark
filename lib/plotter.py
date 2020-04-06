#!/usr/bin/env python3

# Library of plotting functions

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.ticker import MultipleLocator


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
            ax.text(
                x + 0.5,
                y + 0.5,
                '{0:{fmt}}'.format(df.iloc[y, x], fmt=format),
                color="black",
                fontsize=9,
                horizontalalignment='center',
                verticalalignment='center',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.2,
                    'pad': 0
                }
            )
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
    plt.show()


# Plot grid on the axis ax
def drawGrid(ax, xstep=50, ystep=None, minor_ticks_x=5., minor_ticks_y=5.):
    # ax.set_xlim(0, None)
    # ax.set_ylim(0, None)
    minorLocatorX = MultipleLocator(xstep / minor_ticks_x)
    majorLocatorX = MultipleLocator(xstep)
    ax.xaxis.set_major_locator(majorLocatorX)
    ax.xaxis.set_minor_locator(minorLocatorX)
    if ystep is not None:
        minorLocatorY = MultipleLocator(ystep / minor_ticks_y)
        majorLocatorY = MultipleLocator(ystep)
        ax.yaxis.set_minor_locator(minorLocatorY)
        ax.yaxis.set_major_locator(majorLocatorY)
    ax.grid(ls=":", lw=.5, alpha=0.5, color="black")
    ax.grid(which='minor', ls=':', lw=.5, alpha=.3, color='grey')
