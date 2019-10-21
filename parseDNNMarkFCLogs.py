#!/usr/bin/env python
# Plot DNNMark time logs from .log files

import re
import sys
import os
import io
from cycler import cycler
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import LinearLocator
import pandas as pd
# sys.path.insert(0, '/HDD2/ML/mlbenchmarks/DNNMark/')
import lib

import argparse
usage = "python process_chainer_logs.py --text <text>"

parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("--text", "-t", default="", help="Notes to place on the plot")
parser.add_argument("--dir", "-d", default=".", help="Directory with log files")
args = parser.parse_args()

print "Numpy:", np.__version__
print "Pandas:", pd.__version__
print "Matplotlib:", matplotlib.__version__
print "Seaborn:", sns.__version__


def drawGrid(ax, xstep=50, ystep=None):
    ax.grid(ls=":", alpha=.6)
#     ax.set_ylabel("time (s)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    minorLocatorX = MultipleLocator(xstep / 5)
    majorLocatorX = MultipleLocator(xstep)
    ax.xaxis.set_major_locator(majorLocatorX)
    ax.xaxis.set_minor_locator(minorLocatorX)
    if ystep is not None:
        minorLocatorY = MultipleLocator(ystep / 5.)
        majorLocatorY = MultipleLocator(ystep)
        ax.yaxis.set_minor_locator(minorLocatorY)
        ax.yaxis.set_major_locator(majorLocatorY)
    ax.grid(which='minor', linestyle=':', linewidth=.5, alpha=.5)
    ax.grid(which="major", ls=":", alpha=0.25, color="black")


def readLogFiles(logdir, pars):
    dflog = lib.lib.readLogs(logdir, pars, debug=False)
    if dflog is None or dflog.shape[0] < 1:
        print logdir, "No logs!"
        return None
    dflog[["batch", "run"]] = dflog[["batch", "run"]].astype(np.int)
    dflog[["time"]] = dflog[["time"]].astype(np.float)
    # Convert ms to s
    dflog["time"] = dflog["time"] / 1000.
    dflog.sort_values(by=["batch"], inplace=True)
    dflog.reset_index(drop=True, inplace=True)
    return dflog

# Aggerate time group-wise


def group_func(groupdf):
    # print "group"
    # print groupdf
    s = pd.Series()
    s['time'] = groupdf["time"].sum()
    # print s['time']
    return s


output_patterns = [
    re.compile(r"Total running time\(ms\): ([0-9\.\e\+]+)"),
    re.compile(r"NVDRV:([0-9\.\-]+),CUDA:([0-9\.\-]+),cuDNN:([0-9\-\.]+)"),
    re.compile(r"GPU[0-9]+: ([^,]+), ([0-9]+) MiB, ([0-9]+) MiB"),
    re.compile(r"CPU\(s\):\s+(\d+)"),
    re.compile(r"Model name:\s+(.+)"),
    re.compile(r"CPU MHz:\s+([0-9\.]+)"),
    re.compile(r"CPU max MHz:\s+([0-9\.]+)")
]
filename_pattern = re.compile(
    r"^dnnmark_([a-zA-Z0-9@\.]+)_test_([a-z]+)_fc_b([0-9]+)_shape([0-9\-]+)_([0-9]+)\.log$")
columns = ["machine", "direction", "batch", "shape", "run"]
pars = {
    "output_patterns":
    output_patterns,
    "parameters": [
        "time",
        ["NVdrv", "CUDA", "cuDNN"],
        ["GPU model", "GPU memory.total", "GPU memory.free"],
        "CPUs", "CPU model",
        "CPU MHz", "CPU MHz max"
    ],
    "filename_pattern":
    filename_pattern,
    "columns":
    columns
}

logdir = args.dir
print "Reading from", logdir
df_logs = readLogFiles(logdir, pars)

for machine, mgroup in df_logs.groupby(["machine"]):
    print "{}\t:\t{}".format(machine, mgroup.shape[0])

# Drop empty columns
df_logs = df_logs.dropna(axis=1, how='all')

# Check errors
error_logs = df_logs[df_logs.isna().any(axis=1)]
if error_logs.shape[0] > 0:
    print error_logs.shape[0], "errors"
    print error_logs.loc[:, error_logs.isna().any(axis=0)]
    print "---"
    print error_logs
    print "---"

clean_logs = df_logs[~df_logs.index.isin(error_logs.index)].copy()  # Avoid warning "value set on a copy"
print "Errors shape", error_logs.shape
print "Clean shape ", clean_logs.shape

# Average time beteween runs
runs = max(clean_logs["run"].unique())
if runs > 1:
    print "Get average time between runs"
    groupcolumns = list(set(clean_logs.columns) - set(["run", "time"]))
    clean_logs = clean_logs.groupby(groupcolumns, as_index=False).agg(["mean", "max", "min"])
    clean_logs.reset_index(inplace=True)
    clean_logs.drop(["run"], axis=1, level=0, inplace=True)
    clean_logs.columns = [col[0] if col[1] == "" else col[1] for col in clean_logs.columns.values]
    clean_logs.rename({"mean": "time"}, axis="columns", inplace=True)

    print clean_logs.head(2)


fix_missing_cuda = False
if fix_missing_cuda:
    nvdrv, cuda, cudnn = clean_logs.loc[0, ["NVdrv", "CUDA", "cuDNN"]].values
    print "Reuse these info for missing data: nvdrv:{} cuda:{} cudnn:{}".format(nvdrv, cuda, cudnn)
    clean_logs.loc[(clean_logs.index.isin(error_logs.index)), ["NVdrv", "CUDA", "cuDNN"]] = [nvdrv, cuda, cudnn]

clean_logs["env"] = clean_logs["machine"].map(str) + "\n"
if "GPU model" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + clean_logs["GPU model"].map(str) + "  "
if "GPU memory.total" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + clean_logs["GPU memory.total"].map(str) + "  MiB\n"
if "NVdrv" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + "NVDRV:" + clean_logs["NVdrv"].map(str) + ","
if "CUDA" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + "CUDA:" + clean_logs["CUDA"].map(str) + ","
if "cuDNN" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + "cuDNN:" + clean_logs["cuDNN"].map(str) + "\n"
if "CPUs" in clean_logs.columns:
    clean_logs["env"] = clean_logs["env"] + clean_logs["CPUs"].map(str) + "x " + \
        clean_logs["CPU model"].map(str) + "(" + clean_logs["CPU MHz max"].map(str) + ")"

clean_logs["shape"] = clean_logs["shape"].str.replace("-", "_")

# Remove algo groups with errors
clean_logs = clean_logs[["env", "machine", "shape", "direction", "batch", "time"]]
# Check number of samples in machine-shape-algos groups
print "Check number of samples in machine-shape-algos groups"
groupdf = clean_logs.groupby(["env", "machine", "shape"], as_index=False).count()
print groupdf.query("batch < 29")
# Get all groups with not enough data
missing_data = groupdf.query("batch < 29")
if missing_data.shape[0] > 0:
    print "Missing data:"
    print missing_data
    print "---"
    merged = clean_logs.merge(missing_data, on=["env", "machine", "shape",
                                                "algofwd", "algo", "algod"], how="left", indicator=True)
    print "Merged:"
    print merged.head()
    print "---"
    clean_logs = clean_logs[merged["_merge"] == "left_only"]
    print "removed missing from clean_logs"
    print "dnnmark algos:"
    algogroups = clean_logs.groupby(["algofwd", "algo", "algod"])
    algogroups.groups.keys()
    print clean_logs.head()
    # Check that no missing data left
    groupdf = clean_logs.groupby(["env", "machine", "shape", "algofwd", "algo", "algod"], as_index=False).count()
    # Get all groups with not enough data
    missing_data = groupdf.query("batch<6")
    print "Missing data (should be empty):"
    print missing_data
    print "---"
print "clean_logs"
print clean_logs.sample(n=4)

# Save clean logs
machines = "_".join(clean_logs["machine"].unique())
csv_file = os.path.join(logdir, "dnnmark_logs_{}.csv".format(machines))
clean_logs.to_csv(csv_file, index=False)
print ("CSV saved to {}".format(csv_file))


# Plot time per shape
for machine in clean_logs["machine"].unique():
    print "Machine: {}".format(machine)
    mlogs = clean_logs[clean_logs["machine"] == machine]
    print "Environment:"
    environment = mlogs.iloc[0]["env"]
    print environment
    fg = sns.FacetGrid(mlogs.sort_values(by=["batch"]), col="shape", hue="direction",
                       height=1.9, aspect=2.3, margin_titles=True, sharey=False)
    if runs > 1:
        # Fill between min and max
        g = fg.map(plt.fill_between, "batch", "max", "min", color="red", alpha=0.5)
    g = fg.map(plt.plot, "batch", "time", lw=1, alpha=1, ms=4, marker="o",
               fillstyle="full", markerfacecolor="#ffffff").add_legend()

    plt.subplots_adjust(top=0.7)
    g.fig.suptitle("DNNMark time (s) on {}".format(machine), fontsize=24)

    for ax_arr in np.nditer(fg.axes, flags=["refs_ok"]):
        ax = ax_arr.item()
        drawGrid(ax, xstep=50)

    [plt.setp(axg.texts, text="") for axg in g.axes.flat]
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=12)
    axes = g.axes
    for i, row in enumerate(axes):
        #     print row.shape
        for j, column in enumerate(row):
            if j == 0:
                ax = axes[i, j]
                ax.set_ylabel("time (s)", fontsize=12)
            if i == axes.shape[0] - 1:
                ax = axes[i, j]
                ax.set_xlabel("mini-batch size", fontsize=12)

    cwd = os.getcwd()
    logdir_path = os.path.join(cwd, logdir).split("DNNMark/")[1]

    # mini-batch sizes
    batchsizes = " ".join([str(a) for a in sorted(clean_logs["batch"].unique())])
    text = logdir_path + "\n" + batchsizes
    g.fig.text(0, 0, text, ha="left", va="top", fontsize=8)

    x = 0.8  # bottom text horizontal position
    g.fig.text(x, 0, environment, ha="left", va="top", fontsize=8)
    x = 0.45
    if args.text:
        text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
        g.fig.text(x, -0.1, text_, ha="left", va="top", fontsize=8)
        x += 0.15

    readme_path = os.path.join(logdir, "README")
    if os.path.exists(readme_path):
        family = "monospace"
        with io.open(readme_path, mode="r", encoding='utf-8') as f:
            text_ = f.read()
        g.fig.text(x, -0.1, text_, ha="left", va="top", fontsize=12, family=family)
        x += 0.2

    # Print errors on the plot
    mfont = {'family': 'monospace'}
    if error_logs.shape[0] > 0:
        error_logs_ = error_logs[["machine", "shape", "batch"]].drop_duplicates()
        text_ = "Errors\n" + error_logs_.to_string()
        g.fig.text(x, -0.1, text_, ha="left", va="top", fontsize=8, **mfont)

    fig_file = os.path.join(logdir, "DNNMark_shape_times_{}.pdf".format(machine))
    plt.savefig(fig_file, bbox_inches="tight")
    print "Saved plot to", fig_file
    plt.close()

# Calculate total time
print "Calculate total time"
clean_logs["shape"] = clean_logs["shape"].str.replace("-", "_")

aggregate_columns = ["env", "machine", "batch"]
df_ = clean_logs[aggregate_columns + ["time", "direction", "shape"]]
print df_.head()
dnnmark_aggr = df_.groupby(by=aggregate_columns, sort=False).apply(group_func)
dnnmark_aggr.reset_index(inplace=True)
# dnnmark_aggr.drop(["tmp"], axis=1, inplace=True)
print "Aggrgated df"
print dnnmark_aggr.head()

if 'algos' in clean_logs.columns:
    fg = sns.FacetGrid(dnnmark_aggr, col="env", hue="algos",
                       height=5, aspect=1.7, margin_titles=True, sharey=False)
else:
    fg = sns.FacetGrid(dnnmark_aggr, col="env",
                       height=5, aspect=1.7, margin_titles=True, sharey=False)
g = fg.map(plt.plot, "batch", "time", lw=1, alpha=1, ms=4, marker="o",
           fillstyle="full", markerfacecolor="#ffffff").add_legend()
plt.subplots_adjust(top=0.75)
g.fig.suptitle("DNNMark FC layers time (s)", fontsize=16)

for ax_arr in np.nditer(fg.axes, flags=["refs_ok"]):
    ax = ax_arr.item()
    drawGrid(ax, xstep=50)

[plt.setp(gax.texts, text="") for gax in g.axes.flat]
g.set_titles(row_template='{row_name}', col_template='{col_name}', size=12)

axes = g.axes
for i, row in enumerate(axes):
    #     print row.shape
    for j, column in enumerate(row):
        if j == 0:
            ax = axes[i, j]
            ax.set_ylabel("time (s)", fontsize=12)
        if i == axes.shape[0] - 1:
            ax = axes[i, j]
            ax.set_xlabel("mini-batch size", fontsize=12)
#         ax.set_title("")

if args.text:
    text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
    g.fig.text(0.1, 0, text_, ha="left", va="top", fontsize=8)

fig_file = os.path.join(logdir, "FC_time_{}.pdf".format(machines))
plt.savefig(fig_file, bbox_inches="tight")
print "Saved plot to", fig_file
plt.close()

# Save VGG time
print "FC aggregated data"
csv_file = os.path.join(logdir, "FC_time_{}.csv".format(machines))
dnnmark_aggr.to_csv(csv_file, index=False)
print ("CSV saved to {}".format(csv_file))
