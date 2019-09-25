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
# sys.path.insert(0, '../../../')
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
            print "No data for {}".format(shape)
            return 0
        except TypeError as e:
            print "Error converting", series[shape], "to float"
            print series[:2]
            print "shape:", shape
            print e
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
    re.compile(r"CPU max MHz:\s+([0-9\.]+)"),
    re.compile(r"FWD conv. algo: (\d)"),
    re.compile(r"BWD conv. Filter algo: (\d)"),
    re.compile(r"BWD conv. Data algo: (\d)"),
    re.compile(r"ConvFwd_1: ([0-9\.\e\+]+)ms"),
    re.compile(r"ConvBwdFilter_1: ([0-9\.\e\+]+)ms"),
    re.compile(r"ConvBwdData_1: ([0-9\.\e\+]+)ms")
]
filename_pattern = re.compile(
    r"^dnnmark_([a-zA-Z0-9@\.]+)_convolution_block_shape([0-9\-]+)_bs([0-9]+)_algos([a-z0-9]+)-([a-z0-9]+)-([0-9a-z]*)_([0-9]+)\.log$")
columns = ["machine", "shape", "batch", "algofwd", "algo", "algod", "run"]
pars = {
    "output_patterns":
    output_patterns,
    "parameters": [
        "time", ["NVdrv", "CUDA", "cuDNN"], ["GPU model", "GPU memory.total", "GPU memory.free"], "CPUs", "CPU model",
        "CPU MHz", "CPU MHz max", "fwdalgo", "bwdalgo", "bwdalgodata", "ConvFwd_time", "ConvBwdFilter_time",
        "ConvBwdData_time"
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
    print error_logs.shape[0], "errors"
    print error_logs.loc[:, error_logs.isna().any(axis=0)]

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

clean_logs["env"] = clean_logs["machine"].map(str) + "\n" + \
    clean_logs["GPU model"].map(str) + "  " + clean_logs["GPU memory.total"].map(str) + " MiB\nNVDRV:" + \
    clean_logs["NVdrv"].map(str) + ", CUDA" + clean_logs["CUDA"].map(str) + \
    ", cuDNN" + clean_logs["cuDNN"].map(str) + "\n" + clean_logs["CPUs"].map(str) + "x " + \
    clean_logs["CPU model"].map(str) + "(" + clean_logs["CPU MHz max"].map(str) + ")"
clean_logs["shape"] = clean_logs["shape"].str.replace("-", "_")

# Remove algo groups with errors
clean_logs = clean_logs[["env", "machine", "shape", "algofwd", "algo", "algod", "batch", "time"]]
# Check number of samples in machine-shape-algos groups
print "Check number of samples in machine-shape-algos groups"
groupdf = clean_logs.groupby(["env", "machine", "shape", "algofwd", "algo", "algod"], as_index=False).count()
# Get all groups with not enough data
missing_data = groupdf.query("batch < 6")
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
print clean_logs.head()

# Save clean logs
machines = "_".join(clean_logs["machine"].unique())
csv_file = "dnnmark_logs_{}.csv".format(machines)
clean_logs.to_csv(csv_file, index=False)
print ("CSV saved to {}".format(csv_file))

clean_logs["back_algos"] = clean_logs["algo"].map(
    str) + "_" + clean_logs["algod"].map(str)
# Plot time per shape
for machine in clean_logs["machine"].unique():
    print "Machine: {}".format(machine)
    mlogs = clean_logs[clean_logs["machine"] == machine]
    print "Environment:"
    environment = mlogs.iloc[0]["env"]
    print environment
    fg = sns.FacetGrid(mlogs.sort_values(by=["batch"]), row="back_algos", col="shape", hue="algofwd",
                       height=1.9, aspect=2.3, margin_titles=True, sharey=False)
    if runs > 1:
        # Fill between min and max
        g = fg.map(plt.fill_between, "batch", "max", "min", color="red", alpha=0.5)
    g = fg.map(plt.plot, "batch", "time", lw=1, alpha=1, ms=4, marker="o",
               fillstyle="full", markerfacecolor="#ffffff").add_legend()

    plt.subplots_adjust(top=0.9)
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
    logdir_path = cwd.split("DNNMark/")[1]
    g.fig.text(0, 0, logdir_path, ha="left", va="top", fontsize=12)
    x = 0.2  # bottom text horizontal position
    g.fig.text(x, 0, environment, ha="left", va="top", fontsize=12)
    x = 0.35
    if args.text:
        text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
        g.fig.text(x, 0, text_, ha="left", va="top", fontsize=12)
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
        error_logs_ = error_logs[["machine", "shape", "batch"]].drop_duplicates()
        text_ = "Errors\n" + error_logs_.to_string()
        g.fig.text(x, 0, text_, ha="left", va="top", fontsize=12, **mfont)

    fig_file = "DNNMark_shape_times_{}.pdf".format(machine)
    plt.savefig(fig_file, bbox_inches="tight")
    print "Saved plot to", fig_file
    plt.close()
