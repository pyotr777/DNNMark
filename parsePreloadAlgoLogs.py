#!/usr/bin/env python3

# Read algroithms from cudnn_log.csv file produced with PRELOADed cudnn functions.

# 2019 (C) Peter Bryzgalov @ CHITECH Stair Lab

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from functools import partial
# from pandarallel import pandarallel

# # Initialize pandarallel
# pandarallel.initialize()

parser = argparse.ArgumentParser()
parser.add_argument("--dir", "-d", default=".", help="Directory with log files")
parser.add_argument("--file", "-f", default="cudnn_log.csv", help="Name of CSV file with preload_algo logs")
parser.add_argument("--framework", default="TensorFlow", help="Software framework name to put into plot titles")
parser.add_argument("--autotune", '-a', action="store_true", help="ML framework uses cuDNN autotuning")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Add number of convolutions with a particular configuration to the plot annotations"
)
args = parser.parse_args()

print("Numpy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)


# Parse group of consecutive rows with the same batch and shape
# Return one row (series)
# For TensorFlow logs (or other ML framework using cuDNN autotuning )
# Use only the last line from the group of consequitive lines with the same values (except for algorithm numbers).
def parseConsecutiveAlgoRows(grp, autotune=False):
    series = pd.Series()
    series['batch'] = grp['batch'].iloc[0]
    series['shape'] = grp['shape'].iloc[0]
    series['function'] = grp['function'].iloc[0]
    series['group'] = grp['group'].iloc[0]
    series['count'] = grp['algo'].count()
    if autotune:
        series['algo'] = grp['algo'].iloc[-1]  # algorthm from the last row
    else:
        series['algo'] = grp['algo'].min()
        series['max'] = grp['algo'].max()

        if series['algo'] != series['max']:
            print("Altering algorithms in the group")
            print(grp)
    return series


# Return tuple of algorithms:
# FWD, BWD fileter, BWD data
# for the given batch and shape
def getAlgos(df, batch, shape):
    shape_ = "_".join([str(s) for s in shape])
    # Select relevant rows from DF
    selected = df.query("batch==@batch & shape==@shape_")
    if selected.shape[0] > 1:
        print("Duplicate entries in DF for bs/shape {} / {}".format(batch, shape_))
        print(selected)
        return None
    if selected.shape[0] == 0:
        return None, None, None

    fwd = selected["fwdalgo"].values[0]
    bwd_f = selected["algo"].values[0]
    bwd_d = None
    try:
        bwd_d = selected["algod"].values[0]
    except Exception as e:
        print("No bwd data algo in {}".format(selected.shape))
        print(e)

    if selected.isna().any(axis=1).values[0]:
        print("{}".format(selected))
    return fwd, bwd_f, bwd_d


# Return a series with FWD, BWD fileter, BWD data
# for the given batch and shape
def getAlgoSeries(data, autotune=False, groups=None):
    # print("DF in getAlgoSeries")
    # print(data.head())
    # print("DF shape {}".format(data.shape))
    df_ = data.copy()
    # Squash consequative rows with same values
    group = df_.iloc[0]["group"]
    df_ = df_.groupby("group", as_index=False, sort=False).apply(parseConsecutiveAlgoRows, autotune=autotune)
    # # Remove groups made from more than 3 consequative rows - VGG16 has up to 3 same layers
    # df_ = df_[df_['count'] <= 3]
    df_.drop(['count'], axis=1, inplace=True)
    if "max" in df_.columns:
        df_.drop(['max'], axis=1, inplace=True)
    # print('grouped')
    # print(df_.head())
    # print(type(df_), df_.shape)
    # Pivot function column into columns: fwdalgo, algo and algod
    df_ = df_.pivot_table(index=['batch', 'shape'], columns=["function"], values="algo")
    df_.reset_index(inplace=True)
    df_.columns.name = None
    # print('Ready')
    print("{}/{} {}".format(group, groups, df_.iloc[0].values))
    # print('columns:', df_.columns)
    # print('index:  ', df_.index)
    if df_.shape[0] > 1:
        print("Duplicate entries in DF for bs/shape {}".format(df_[['batch', 'shape']]))
        print(df_)
        return None

    return df_


# Squash groups with consecuative batch and shape,
# pivot function column to FWD, BWD filter, BWD data columns.
def extractAlgoColumns(df, autotune=False):
    df_ = df.copy()
    # Squash consequative rows with same values
    df_["distinct"] = df_["batch"].astype(str) + "/" + df_["shape"] + "/" + df_["function"]
    df_["group"] = (df_["distinct"] != df_["distinct"].shift(1)).cumsum()
    groups = df_["group"].max()
    try:
        df_ = df_.groupby(["group"], as_index=False, sort=False).apply(getAlgoSeries, autotune=autotune, groups=groups)
    except Exception as e:
        print(df_.head())
        print(e)
    # if "algod" in df_.columns:
    #     df_["algod"] = df_["algod"].astype(pd.Int64Dtype())
    return df_


# Parse list of algo numbers for display
# If algo not changing returns 0.
def minmax(l):
    smin = int(min(l))
    smax = int(max(l))
    return smax - smin


# Returned list of values
def concat_values(a, debug=False):
    s = "/".join([r"{:.0f}".format(s) for s in a.unique()])
    if debug:
        l = len(a)
        s = r"${}_{{{}}}$".format(s, l)
    return s


# Returns mean of input list values
def mean(l):
    intl = [int(l1) for l1 in l]
    return int(np.mean(intl))


# Read algorithms
path = os.path.join(args.dir, args.file)
print("Reading from {}".format(path))
df = pd.read_csv(path, header=None, names=["batch", "shape", "function", "algo"])
print(df.tail())
df["batch"] = df["batch"].astype(int)
# df = df[df['batch'] >= 7]
print("Logs shape: {}".format(df.shape))
batchsizes = df["batch"].unique()
print(batchsizes)
print(df["function"].unique())
print("Autotunning is {}".format(args.autotune))
dfparsed = extractAlgoColumns(df, args.autotune)
mbs = dfparsed["batch"].unique()
l = len(mbs)
print("parsed {}mbs: {}".format(l, " ".join([str(b) for b in sorted(mbs)])))
algos = [algo for algo in ['algo', 'algod', 'fwdalgo'] if algo in dfparsed.columns]
extra_plotwidth = l * 0.18 * len(algos)
fig, axs = plt.subplots(1, 3, figsize=(12 + extra_plotwidth, 6))
for i, column in enumerate(algos):
    ax = axs[i]
    # Select rows with relevant algos
    df_ = dfparsed[dfparsed[column].notna()]
    print(column)
    print(df_.head(2))
    df_diff = df_.pivot_table(index="shape", columns="batch", values=column, aggfunc=minmax)
    dflabels = df_.pivot_table(
        index="shape", columns="batch", values=column, aggfunc=partial(concat_values, debug=args.debug)
    )
    dfvalues = df_.pivot_table(index="shape", columns="batch", values=column, aggfunc=mean)
    # Display all rows having not 0.
    # Not 0 means there are variations in algo number for this shape (row in DF)
    alter_algo = df_diff[df_diff.ne(0).any(axis=1)]
    if alter_algo.shape[0] > 0:
        print("Changing algorithm for {}".format(column))
        # print(only rows and columns with changed algos)
        print(dflabels[df_diff.ne(0).any(axis=1)])
    annot = dflabels.values
    fmt = 's'
    try:
        annot = annot.astype(int)
        fmt = 'd'
    except:
        print("Type of {} cannot be converted to int".format(annot.dtype))
        pass
    print("Annotations type:{} fmt:{}".format(annot.dtype, fmt))
    sns.heatmap(dfvalues, annot=annot, annot_kws={"size": 8}, xticklabels=True, fmt=fmt, vmin=0, vmax=7, ax=ax)
    ax.set_title("{} {}".format(args.framework, column), fontsize=16)
    ax.tick_params(axis="x", labelsize=10)
    if i != 0:
        ax.set_ylabel("")

plt.tight_layout()
fig_file = os.path.join(args.dir, "algo_logs.pdf")
plt.savefig(fig_file, bbox_inches="tight")
print("Saved plot to", fig_file)
plt.close()
