#!/usr/bin/env python3

# Run DNNMark with same convolution algorithms as Chainer v4.0.0.
# Aglroithms read from chainer_algos_<host>.csv file produced with PRELOADed cudnn functions.
# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import numpy as np
import time
import os
import datetime
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Parse group of consecutive rows with the same batch and shape
# Return one row (series)
def parseConsecutiveAlgoRows(grp):
    series = pd.Series()
    series['batch'] = grp['batch'].iloc[0]
    series['shape'] = grp['shape'].iloc[0]
    series['function'] = grp['function'].iloc[0]
    series['algo'] = grp['algo'].min()
    series['max'] = grp['algo'].max()
    series['count'] = grp['algo'].count()
    series['group'] = grp['group'].iloc[0]
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
def getAlgoSeries(data):
    # print("Data")
    # print(data.head())
    # print(data.shape)
    df_ = data.copy()
    # Squash consequative rows with same values
    df_["group"] = (df_["function"] != df_["function"].shift(1)).cumsum()
    df_ = df_.groupby("group", as_index=False, sort=False).apply(parseConsecutiveAlgoRows)

    # Remove groups made from more than 3 consequative rows - VGG16 has up to 3 same layers
    df_ = df_[df_['count'] <= 3]
    df_.drop(['max', 'count'], axis=1, inplace=True)
    # print('grouped')
    # print(df_.head())
    # print(type(df_), df_.shape)
    # Pivot function column into columns: fwdalgo, algo and algod
    df_ = df_.pivot_table(index=['batch', 'shape'], columns=["function"], values="algo")
    df_.reset_index(inplace=True)
    df_.columns.name = None
    # print('Ready')
    # print(df_)
    # print('columns:', df_.columns)
    # print('index:  ', df_.index)
    if df_.shape[0] > 1:
        print("Duplicate entries in DF for bs/shape {}".format(df_[['batch', 'shape']]))
        print(df_)
        return None

    return df_


# Squash groups with consecuative batch and shape,
# pivot function column to FWD, BWD filter, BWD data columns.
def extractAlgoColumns(df):
    df_ = df.copy()
    # Squash consequative rows with same values
    df_ = df_.groupby(["batch", "shape"], as_index=False, sort=False).apply(getAlgoSeries)
    df_["algod"] = df_["algod"].astype(pd.Int64Dtype())
    return df_


# Parse list of algo numbers for display
# If algo not changing returns 0.
def minmax(l):
    smin = int(min(l))
    smax = int(max(l))
    return smax - smin


# Returned list of values
def concat_values(l):
    s = "/".join([str(int(s)) for s in l.unique()])
    return s


# Returns mean of input list values
def mean(l):
    intl = [int(l1) for l1 in l]
    return int(np.mean(intl))


# Read algorithms
df = pd.read_csv("chainer_cudnn_logs.csv", header=None, names=["batch", "shape", "function", "algo"])
print(df.tail())
df["batch"] = df["batch"].astype(int)
df = df[df['batch'] >= 7]
print(df.shape)
batchsizes = df["batch"].unique()
print(batchsizes)
print(df["function"].unique())
dfparsed = extractAlgoColumns(df)
print("parsed")
print(dfparsed.head())
fig, axs = plt.subplots(1, 3, figsize=(24, 6))
for i, column in enumerate(['algo', 'algod', 'fwdalgo']):
    ax = axs[i]
    # Select rows with relevant algos
    df_ = dfparsed[dfparsed[column].notna()]
    print(df_.head())
    df_diff = df_.pivot_table(index="shape", columns="batch", values=column,
                              aggfunc=minmax)
    dflabels = df_.pivot_table(index="shape", columns="batch", values=column,
                               aggfunc=concat_values)
    dfvalues = df_.pivot_table(index="shape", columns="batch", values=column,
                               aggfunc=mean)
    # Display all rows having not 0.
    # Not 0 means there are variations in algo number for this shape (row in DF)
    alter_algo = df_diff[df_diff.ne(0).any(axis=1)]
    if alter_algo.shape[0] > 0:
        print("Changing algorithm for {}".format(column))
        # print(only rows and columns with changed algos)
        print(dflabels[df_diff.ne(0).any(axis=1)])
    annot = dflabels.astype(str)
    ax = sns.heatmap(dfvalues, annot=annot, annot_kws={"size": 6},
                     xticklabels=True, fmt="s", vmin=0, vmax=6, ax=ax)
    ax.set_title("Chainer {}".format(column), fontsize=16)
    ax.tick_params(axis="x", labelsize=6)
    if i != 0:
        ax.set_ylabel("")

fig_file = "algo_logs_{}.pdf".format("mouse")
plt.savefig(fig_file, bbox_inches="tight")
print("Saved plot to", fig_file)
plt.close()

# Set GPU range
gpus = list(range(0, 1))

# Change hostname
host = "mouse"

# Set number of runs
runs = 1

# Set mini-batch sizes
# batchsizes = [7, 8, 9] + range(10, 200, 10) + range(200, 501, 50)
# Log algos
# batchsizes = [10, 20, 50, 100, 150, 200, 400, 500]

# VGG model convolution shapes
configs = [(2, 512, 512), (4, 512, 512), (4, 256, 512), (8, 256, 256),
           (8, 128, 256), (16, 128, 128), (16, 64, 128), (32, 64, 64), (32, 3, 64)]

benchmark = "test_composed_model"
default_benchmark = "test_composed_model"

datasetsize = 50000
date = datetime.datetime.today().strftime('%Y%m%d')
nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []

# other_options = " --bwd_filter_pref no_workspace "
other_options = ""

if benchmark != default_benchmark:
    command = "./run_dnnmark_template.sh{} -b {}".format(other_options, benchmark)
else:
    command = "./run_dnnmark_template.sh{}".format(other_options)

if "/" in benchmark:
    benchmark = benchmark.split("/")[-1]
logdir = "logs/{}/dnnmark_{}_microseries_{}/".format(host, benchmark, date)

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)
logfile_base = "dnnmark_{}_{}".format(host, benchmark)
for config in configs:
    imsize, channels, conv = config
    for batch in batchsizes:
        iterations = int(math.ceil(datasetsize / batch))
        # print("BS: {}, Iterations: {}".format(batch, iterations))
        algofwd, algo, algod = getAlgos(dfparsed, batch, config)
        if algofwd is None:
            print("No FWD algo for {} {}".format(batch, config))
            continue

        # Use default algod if not logged
        algod_option = ""
        if not math.isnan(algod):
            algod_option = "--algod " + str(algod)
        else:
            algod = "cudnn"
            algod_option = "--algod cudnn"
        algofwd_option = ""
        if algofwd is not None:
            algofwd_option = "--algofwd " + str(algofwd)
        else:
            algofwd = ""

        # print("FWD {}, BWD data {}, BWD filter {}".format(algofwd, algod, algo))
        logname = "{}_shape{}-{}-{}_bs{}_algos{}-{}-{}".format(
            logfile_base, imsize, channels, conv, batch, algofwd, algo, algod)
        print("formed :", logname)
        for run in range(runs):
            logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
            if os.path.isfile(logfile):
                print("file", logfile, "exists.")
            else:
                command_pars = command + " -c {} -n {} -k {} -w {} -h {} --algo {} {} {} -d {}{} --warmup 1".format(
                    channels, batch, conv, imsize, imsize, algo, algod_option, algofwd_option, datasetsize, debuginfo_option)
                task = {"comm": command_pars, "logfile": logfile,
                        "batch": batch, "conv": conv, "nvsmi": with_memory}
                tasks.append(task)
        if nvprof:
            iterations = 10
            nvlogname = "{}_iter{}".format(logname, iterations)
            command_pars = command + " -c {} -n {} -k {} -w {} -h {} --algo {} --algod {} --algofwd {} --iter {} --warmup 0".format(
                channels, batch, conv, imsize, imsize, algo, algod, algofwd, iterations)
            logfile = os.path.join(logdir, "{}_%p.nvprof".format(nvlogname))
            if os.path.isfile(logfile):
                print("file", logfile, "exists.")
            else:
                profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --csv --log-file {} {}".format(
                    logfile, command_pars)
                task = {"comm": profcommand, "logfile": logfile,
                        "batch": batch, "conv": conv, "nvsmi": False}
                tasks.append(task)

print("Have", len(tasks), "tasks")

gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec3.getNextFreeGPU(gpus, start=gpu + 1, c=3, d=2, nvsmi=tasks[i]["nvsmi"], mode="dmon", debug=False)
    gpu_info = multigpuexec3.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"], "w+")
    f.write(tasks[i]["comm"] + "\n")
    f.write("b{} conv{}\n".format(tasks[i]["batch"], tasks[i]["conv"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.close()
    print(time.strftime("[%d,%H:%M:%S]"))
    multigpuexec3.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print(tasks[i]["logfile"])
    print("{}/{} tasks".format(i + 1, len(tasks)))
    time.sleep(0)
