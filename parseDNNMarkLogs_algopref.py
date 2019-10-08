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


def shape_sum_column(series):
    # print "series in shape_sum_column\n", series
    VGG_shapes = {
        "32_3_64": 1,
        "32_64_64": 1,
        "16_64_128": 1,
        "16_128_128": 1,
        "8_128_256": 1,
        "8_256_256": 2,
        "4_256_512": 1,
        "4_512_512": 2,
        "2_512_512": 3}
    sum_ = 0
    for shape, value in VGG_shapes.iteritems():
        try:
            time = float(series[shape])
        except KeyError:
            print "no data for {},".format(shape),
            return 0
        except TypeError as e:
            print "Error converting", series[shape], "to float"
            print series[:2]
            print "shape:", shape
            print e
            sys.exit(1)
            continue
        multipl = float(value)
        sum_ += time * multipl
    return sum_


# Aggerate time group-wise
def group_func(groupdf):
    df_ = groupdf.set_index("shape")
    SUM_time = pd.DataFrame(data=df_[["time"]].apply(shape_sum_column, axis=0).values, columns=["VGG time"])
    SUM_time.index.name = 'tmp'
    return SUM_time


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
    r"^dnnmark_([a-zA-Z0-9@\.]+)_test_composed_model_([a-z\-]+)_shape([0-9\-]+)_bs([0-9]+)_algos([a-zA-Z0-9]+)-([a-zA-Z0-9]+)-([0-9A-Za-z]*)_([0-9]+)\.log$")
columns = ["machine", "algo_pref", "shape", "batch", "algofwd", "algo", "algod", "run"]
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

# Check errors
error_logs = df_logs[df_logs.isna().any(axis=1)]
if error_logs.shape[0] > 0:
    error_logs["algos"] = error_logs["algofwd"].map(
        str) + "_" + error_logs["algo"].map(str) + "_" + error_logs["algod"].map(str)
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
clean_logs["shape"] = clean_logs["shape"].str.replace("-", "_")
clean_logs["algos"] = clean_logs["algofwd"].map(
    str) + "_" + clean_logs["algo"].map(str) + "_" + clean_logs["algod"].map(str)


fix_missing_cuda = False
if fix_missing_cuda:
    nvdrv, cuda, cudnn = clean_logs.loc[0, ["NVdrv", "CUDA", "cuDNN"]].values
    print "Reuse these info for missing data: nvdrv:{} cuda:{} cudnn:{}".format(nvdrv, cuda, cudnn)
    clean_logs.loc[(clean_logs.index.isin(error_logs.index)), ["NVdrv", "CUDA", "cuDNN"]] = [nvdrv, cuda, cudnn]

clean_logs["env"] = clean_logs["machine"].map(str) + "\n" + \
    clean_logs["GPU model"].map(str) + "  " + clean_logs["GPU memory.total"].map(str) + " MiB\nNVDRV:" + \
    clean_logs["NVdrv"].map(str) + ", CUDA" + clean_logs["CUDA"].map(str) + \
    ", cuDNN" + clean_logs["cuDNN"].map(str) + "\n" + clean_logs["CPUs"].map(str) + "x " + \
    clean_logs["CPU model"].map(str) + "(" + clean_logs["CPU MHz max"].map(str) + ")"


# Remove algo groups with errors
clean_logs = clean_logs[["env", "machine", "shape", "algo_pref", "algos", "batch", "time"]]
# Check number of samples in machine-shape-algos groups
print "Check number of samples in machine-shape-algos groups"
groupdf = clean_logs.groupby(["env", "machine", "shape", "algo_pref", "algos"], as_index=False).count()
# Get all groups with not enough data
missing_data = groupdf.query("batch < 6")
if missing_data.shape[0] > 0:
    print "Missing data:"
    print missing_data
    print "---"
    merged = clean_logs.merge(missing_data, on=["env", "machine", "shape", "algo_pref",
                                                "algos"], how="left", indicator=True)
    print "Merged:"
    print merged.head()
    print "---"
    clean_logs = clean_logs[merged["_merge"] == "left_only"]
    print "removed missing from clean_logs"
    print "dnnmark algos:"
    algogroups = clean_logs.groupby(["algos", "algo_pref"])
    algogroups.groups.keys()
    print clean_logs.head()
    # Check that no missing data left
    groupdf = clean_logs.groupby(["env", "machine", "shape", "algos", "algo_pref"], as_index=False).count()
    # Get all groups with not enough data
    missing_data = groupdf.query("batch<6")
    print "Missing data (should be empty):"
    print missing_data
    print "---"
print "clean_logs raw"
print clean_logs.sample(n=4)

# Leave only overlapping batch sizes
batchsizes = []
for name, group in clean_logs.groupby(by=["machine", "shape", "algos"]):
    bs_ = group["batch"].unique()
    if len(batchsizes) == 0:
        batchsizes = bs_
    else:
        batchsizes = [b for b in batchsizes if b in bs_]
print "batchsizes", " ".join(str(bs) for bs in batchsizes)
clean_logs = clean_logs[clean_logs["batch"].isin(batchsizes)]

# Constract fastest series
clean_logs = lib.lib.getLowestFromSeries(clean_logs, group_columns=["env", "machine", "shape", "algos", "batch"],
                                         series="algo_pref", y="time")
print "clean logs with fastest series"
print clean_logs.head(n=9)

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
    fg = sns.FacetGrid(mlogs.sort_values(by=["shape", "batch"]), row="algos", col="shape", hue="algo_pref",
                       height=3, aspect=1.8, margin_titles=True, sharey=False)
    if runs > 1:
        # Fill between min and max
        g = fg.map(plt.fill_between, "batch", "max", "min", color="red", alpha=0.5)
    g = fg.map(plt.plot, "batch", "time", lw=0.5, alpha=0.7, ms=2, marker="o",
               fillstyle="full", markerfacecolor="#ffffff").add_legend()

    plt.subplots_adjust(top=0.6)
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
    g.fig.text(0, 0, text, ha="left", va="top", fontsize=10)

    x = 0.3  # bottom text horizontal position
    g.fig.text(x, 0, environment, ha="left", va="top", fontsize=8)
    x = 0.45
    if args.text:
        text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
        g.fig.text(x, 0, text_, ha="left", va="top", fontsize=8)
        x += 0.15

    readme_path = os.path.join(logdir, "README")
    if os.path.exists(readme_path):
        family = "monospace"
        with io.open(readme_path, mode="r", encoding='utf-8') as f:
            text_ = f.read()
        g.fig.text(x, 0, text_, ha="left", va="top", fontsize=12, family=family)
        x += 0.2

    # Print errors on the plot
    mfont = {'family': 'monospace'}
    if error_logs.shape[0] > 0:
        error_logs_ = error_logs[["machine", "shape", "batch", "algos", "algo_pref"]
                                 ].sort_values(by=["algo_pref", "shape", "batch"]).drop_duplicates()
        text_ = "Errors\n" + error_logs_.to_string()
        g.fig.text(x, 0, text_, ha="left", va="top", fontsize=4, **mfont)

    fig_file = os.path.join(logdir, "DNNMark_shape_times_{}.pdf".format(machine))
    plt.savefig(fig_file, bbox_inches="tight")
    print "Saved plot to", fig_file
    plt.close()

# Calculate VGG time
print "Calculate VGG time"

print clean_logs["algos"].unique()
aggregate_columns = ["env", "machine", "batch", "algos", "algo_pref"]
df_ = clean_logs[aggregate_columns + ["time", "shape"]]
print df_.head()
dnnmark_aggr = df_.groupby(by=aggregate_columns).apply(group_func)
dnnmark_aggr.reset_index(inplace=True)
dnnmark_aggr.drop(["tmp"], axis=1, inplace=True)
print "Aggrgated df algos"
print dnnmark_aggr.head()

fg = sns.FacetGrid(dnnmark_aggr, col="env", hue="algo_pref", row="algos",
                   height=5, aspect=2, margin_titles=True, sharey=False)
g = fg.map(plt.plot, "batch", "VGG time", lw=.6, alpha=.6, ms=3, marker="o",
           fillstyle="full", markerfacecolor="#ffffff").add_legend()
plt.subplots_adjust(top=0.78)
g.fig.suptitle("VGG-aggregated DNNMark time (s)", fontsize=16)

for ax_arr in np.nditer(fg.axes, flags=["refs_ok"]):
    ax = ax_arr.item()
    drawGrid(ax, xstep=50, ystep=50)

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

fig_file = os.path.join(logdir, "VGG_time_{}.pdf".format(machines))
plt.savefig(fig_file, bbox_inches="tight")
print "Saved plot to", fig_file
plt.close()

# Save VGG time
print "VGG-aggregated data"
csv_file = os.path.join(logdir, "VGG_time_{}.csv".format(machines))
dnnmark_aggr.to_csv(csv_file, index=False)
print ("CSV saved to {}".format(csv_file))
