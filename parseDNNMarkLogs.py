#!/usr/bin/env python3
# Plot DNNMark time logs from .log files
# Run with one option in filenames: algoconfig

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
sys.path.insert(0, 'lib/')
import lib3, plotter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", "-t", default="", help="Notes to place on the plot")
parser.add_argument("--dir", "-d", default=".", help="Directory with log files")
parser.add_argument("--noVGG", action="store_true", help="Do not create VGG shape")
parser.add_argument("--singleVGG", action="store_true", default=True,
                    help="Do not discriminate algorithms when aggregating VGG time")
parser.add_argument(
    "--ref", "-r", default="", help="CSV file with reference time series (DF with batch and time columns)"
)
args = parser.parse_args()

print("Numpy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)


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
    dflog = lib3.readLogs(logdir, pars, debug=False)
    if dflog is None or dflog.shape[0] < 1:
        print(logdir, "No logs!")
        return None
    else:
        print("DF shape", dflog.shape)
    dflog[["batch", "run"]] = dflog[["batch", "run"]].astype(np.int)
    dflog[["time"]] = dflog[["time"]].astype(np.float)
    # Convert ms to s
    dflog["time"] = dflog["time"] / 1000.
    dflog.sort_values(by=["batch"], inplace=True)
    dflog.reset_index(drop=True, inplace=True)
    return dflog


def shape_sum_column(series):
    # print("series in shape_sum_column\n", series)
    VGG_shapes = {
        "32_3_64": 1,
        "32_64_64": 1,
        "16_64_128": 1,
        "16_128_128": 1,
        "8_128_256": 1,
        "8_256_256": 2,
        "4_256_512": 1,
        "4_512_512": 2,
        "2_512_512": 3
    }
    sum_ = 0
    for shape, value in VGG_shapes.items():
        try:
            time = float(series[shape])
        except KeyError:
            print("no data for {},".format(shape), )
            return None
        except TypeError as e:
            print("Error converting", series[shape], "to float")
            print(series[:2])
            print("shape:", shape)
            print(e)
            raise

        multipl = float(value)
        sum_ += time * multipl

    return sum_


# Aggerate time group-wise
def group_func(groupdf, agg_columns=["time"]):
    df_ = groupdf[["shape"] + agg_columns].set_index("shape")
    SUM_time = pd.DataFrame()
    for agg_column in agg_columns:
        try:
            SUM_time["VGG " + agg_column] = df_[[agg_column]].apply(shape_sum_column, axis=0).values
            # SUM_time = pd.DataFrame(data=df_[[agg_column]].apply(shape_sum_column, axis=0).values, columns=["VGG time"])
        except Exception as e:
            print("Error applying shape_sum_column function to")
            print(df_[[agg_column]])
            print("---")
            sys.exit(1)
    SUM_time.index.name = 'tmp'
    return SUM_time


output_patterns = [
    re.compile(r"Total running time\(ms\): ([0-9\.e\+]+)"),
    re.compile(r"NVDRV:([0-9\.\-]+),CUDA:([0-9\.\-x]+),cuDNN:([0-9\-\.]+)"),
    re.compile(r"GPU[0-9]+: ([^,]+), ([0-9]+) MiB, ([0-9]+) MiB"),
    re.compile(r"CPU\(s\):\s+(\d+)"),
    re.compile(r"Model name:\s+(.+)"),
    re.compile(r"CPU MHz:\s+([0-9\.]+)"),
    re.compile(r"CPU max MHz:\s+([0-9\.]+)")
]
filename_pattern = re.compile(
    r"^dnnmark_([a-zA-Z0-9@\.]+)[_conf]*_([a-z_]+)_shape([0-9\-]+)_bs([0-9]+)_algos([a-zA-Z0-9\-]+)_([0-9]+)\.log$"
)
columns = ["machine", "config", "shape", "batch", "algos", "run"]
pars = {
    "output_patterns": output_patterns,
    "parameters":
        [
            "time", ["NVdrv", "CUDA", "cuDNN"], ["GPU model", "GPU memory.total", "GPU memory.free"], "CPUs",
            "CPU model", "CPU MHz", "CPU MHz max"
        ],
    "filename_pattern": filename_pattern,
    "columns": columns
}

logdir = args.dir
print("Reading from", logdir)
df_logs = readLogFiles(logdir, pars)

for machine, mgroup in df_logs.groupby(["machine"]):
    print("{}\t:\t{}".format(machine, mgroup.shape[0]))
print(df_logs.head(1))

meta_columns = [
    "config", "NVdrv", "CUDA", "cuDNN", "GPU model", "GPU memory.total", "GPU memory.free", "CPUs", "CPU model",
    "CPU MHz", "CPU MHz max"
]
meta_df = df_logs[meta_columns]
print("Meta rows")
print(meta_df.head(2))

meta_ = meta_df.iloc[0]
meta_rows = int(len(meta_columns) / 2)
meta1 = "\n".join(["{:15s}:{}".format(col, meta_[col]) for col in meta_columns[:meta_rows]])
meta2 = "\n".join(["{:15s}:{}".format(col, meta_[col]) for col in meta_columns[meta_rows:]])
print("Meta data\n{}".format(meta1))

df_logs["env"] = ""
if "GPU model" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + df_logs["GPU model"].map(str) + "  "
if "GPU memory.total" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + df_logs["GPU memory.total"].map(str) + "  MiB\n"
if "NVdrv" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + "NVDRV:" + df_logs["NVdrv"].map(str) + ","
if "CUDA" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + "CUDA:" + df_logs["CUDA"].map(str) + ","
if "cuDNN" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + "cuDNN:" + df_logs["cuDNN"].map(str) + "\n"
if "CPUs" in df_logs.columns:
    df_logs["env"] = df_logs["env"] + df_logs["CPUs"].map(str) + "x " + \
        df_logs["CPU model"].map(str) + "(" + df_logs["CPU MHz max"].map(str) + ")"

# Drop redundunt columns
df_logs = df_logs[["env", "GPU model", "machine", "config", "shape", "algos", "batch", "time", "run"]]
df_logs.rename(columns={"GPU model": "GPU"}, inplace=True)

# Average time beteween runs
print(df_logs["run"].unique())
runs = len(df_logs["run"].unique())
if runs > 1:
    print("Get average time between runs")
    groupcolumns = list(set(df_logs.columns) - set(["run", "time"]))
    df_logs = df_logs.groupby(groupcolumns, as_index=False).agg(["mean", "max", "min"])
    df_logs.reset_index(inplace=True)
    df_logs.drop(["run"], axis=1, level=0, inplace=True)
    df_logs.columns = [col[0] if col[1] == "" else col[1] for col in df_logs.columns.values]
    df_logs.rename({"mean": "time"}, axis="columns", inplace=True)

    print(df_logs.head(2))
df_logs["shape"] = df_logs["shape"].str.replace("-", "_")

# Check errors
error_logs = df_logs[df_logs.isna().any(axis=1)].copy()
if error_logs.shape[0] > 0:
    print(error_logs.shape[0], "errors")
    print(error_logs.loc[:, error_logs.isna().any(axis=0)])
    print("---")
    print(error_logs)
    print("---")

clean_logs = df_logs[~df_logs.index.isin(error_logs.index)].copy()  # Avoid warning "value set on a copy"
print("Errors shape", error_logs.shape)

group_columns = [c for c in clean_logs.columns if c in ["env", "config", "GPU", "machine", "shape", "algos"]]
use_columns = group_columns + [c for c in clean_logs.columns if c in ["batch", "time", "max", "min"]]

clean_logs = clean_logs[use_columns]
print("Clean logs shape {}".format(clean_logs.shape))
# Remove groups with small number of batches (with many errors)
remove_incomplete_series = False
if remove_incomplete_series:
    # Check number of samples in machine-shape-algos-algopref groups
    print("Check number of samples in machine-config-shape-algos groups")
    groupdf = clean_logs.groupby(group_columns, as_index=False).count()
    # Get all groups with not enough data
    missing_data = groupdf.query("batch < 6")
    if missing_data.shape[0] > 0:
        print("Missing data:")
        missing_data = missing_data[group_columns + ["batch"]]
        print(missing_data.shape)
        print(missing_data.drop(["env"], axis=1))  # [["algo_pref", "batch", "shape", "time"]]
        print("---")
        merged = clean_logs.merge(missing_data, on=[group_columns + ["batch"]], how="left", indicator=True)
        print("Merged:")
        print(merged.head())
        print(merged.shape)
        print("---")
        clean_logs = clean_logs[merged["_merge"] == "left_only"]
        print("removed missing from clean_logs")
        print("dnnmark algos:")
        algogroups = clean_logs.groupby(["algos"])
        algogroups.groups.keys()
        print(clean_logs.head())
        # Check that no missing data left
        groupdf = clean_logs.groupby(group_columns, as_index=False).count()
        # Get all groups with not enough data
        missing_data = groupdf.query("batch<6")
        print("Missing data (should be empty):")
        print(missing_data)
        print("---")
        # Leave only overlapping batch sizes
    batchsizes = []
    for name, group in clean_logs.groupby(by=group_columns):
        bs_ = group["batch"].unique()
        if len(batchsizes) == 0:
            batchsizes = bs_
        else:
            batchsizes = [b for b in batchsizes if b in bs_]
    clean_logs = clean_logs[clean_logs["batch"].isin(batchsizes)]

print("clean_logs {}".format(clean_logs.columns))
print(clean_logs.sample(n=3))
print(" ".join([str(x) for x in clean_logs["batch"].unique()]))

construct_fastest = False
if construct_fastest:
    # Constract fastest series
    clean_logs = lib3.getLowestFromSeries(
        clean_logs, group_columns=["GPU", "machine", "shape", "batch"], series="algos", y="time"
    )
    print("clean logs with fastest series")
    print(clean_logs.head(n=4))

# Save clean logs
machine = clean_logs.iloc[0]["machine"]
csv_file = os.path.join(logdir, "dnnmark_logs_{}.csv".format(machine))
clean_logs_ = clean_logs.copy()
clean_logs_["env"] = meta1 + "\n" + meta2
clean_logs_.to_csv(csv_file, index=False)
print(("CSV saved to {}".format(csv_file)))

# Plot time per shape
for machine in clean_logs["machine"].unique():
    print("Machine: {}".format(machine))
    mlogs = clean_logs[clean_logs["machine"] == machine]
    if args.singleVGG:
        fg = sns.FacetGrid(
            mlogs.sort_values(by=["shape", "batch", "algos"]),
            col="shape",
            height=3,
            aspect=1.5,
            margin_titles=True,
            sharey=False
        )
    elif args.noVGG:
        fg = sns.FacetGrid(
            mlogs.sort_values(by=["shape", "batch", "algos"]),
            col="shape",
            row="algos",
            height=3,
            aspect=1.5,
            margin_titles=True,
            sharey=False
        )
    else:
        fg = sns.FacetGrid(
            mlogs.sort_values(by=["shape", "batch", "algos"]),
            col="shape",
            hue="algos",
            height=3,
            aspect=1.5,
            margin_titles=True,
            sharey=False
        )
    if runs > 1:
        # Fill between min and max
        g = fg.map(plt.fill_between, "batch", "max", "min", color="red", alpha=0.5)
    g = fg.map(
        plt.plot, "batch", "time", lw=0.5, alpha=0.7, ms=2, marker="o", fillstyle="full", markerfacecolor="#ffffff"
    ).add_legend()

    for ax_arr in np.nditer(fg.axes, flags=["refs_ok"]):
        ax = ax_arr.item()
        drawGrid(ax, xstep=50)

    [plt.setp(axg.texts, text="") for axg in g.axes.flat]
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=9)
    axes = g.axes
    for i, row in enumerate(axes):
        #     print(row.shape)
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

    x = 0.  # bottom text horizontal position
    y = -.11
    g.fig.text(x, y, meta1, ha="left", va="top", fontsize=7, fontfamily="monospace")
    x += 0.1
    g.fig.text(x, y, meta2, ha="left", va="top", fontsize=7, fontfamily="monospace")
    x += 0.2
    if args.text:
        text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
        g.fig.text(x, y, text_, ha="left", va="top", fontsize=8)
        x += 0.15

    readme_path = os.path.join(logdir, "README")
    if os.path.exists(readme_path):
        family = "monospace"
        with io.open(readme_path, mode="r", encoding='utf-8') as f:
            text_ = f.read()
        g.fig.text(x, y, text_, ha="left", va="top", fontsize=7, family=family)
        x += 0.2

    # print(errors on the plot)
    mfont = {'family': 'monospace'}
    if error_logs.shape[0] > 0:
        error_logs_ = error_logs[["machine", "shape", "batch",
                                  "algos"]].sort_values(by=["algos", "shape", "batch"]).drop_duplicates()
        text_ = "Errors\n" + error_logs_.to_string()
        g.fig.text(x, y, text_, ha="left", va="top", fontsize=8, **mfont)

    fig_file = os.path.join(logdir, "DNNMark_shape_times_{}.pdf".format(machine))
    # plt.subplots_adjust(top=top)
    plt.suptitle("DNNMark time (s) on {}".format(machine), fontsize=18, va="bottom", y=1.01)
    plt.tight_layout()
    plt.savefig(fig_file, bbox_inches="tight")
    print("Saved plot to", fig_file)
    plt.close()

if args.noVGG:
    print("VGG aggregation is disabled with CLI option")
    sys.exit(0)

# Calculate VGG time
print("Calculate VGG time")

aggregate_columns = []
if args.singleVGG:
    aggregate_columns = ["env", "config", "machine", "GPU", "batch"]
else:
    print(clean_logs["algos"].unique())
    aggregate_columns = ["env", "config", "machine", "GPU", "batch", "algos"]
print("Clean logs columns: {}".format(clean_logs.columns.values))
if "max" in clean_logs.columns.values:
    df_ = clean_logs[aggregate_columns + ["time", "max", "min", "shape"]]
else:
    df_ = clean_logs[aggregate_columns + ["time", "shape"]]
print(df_.head())
print("Shape of df_:", df_.shape)
if "max" in clean_logs.columns.values:
    dnnmark_aggr = df_.groupby(by=aggregate_columns).apply(group_func, agg_columns=["time", "max", "min"])
else:
    dnnmark_aggr = df_.groupby(by=aggregate_columns).apply(group_func, agg_columns=["time"])
print("VGG aggregated")
# print(dnnmark_aggr.columns)
# print(dnnmark_aggr)
dnnmark_aggr.reset_index(inplace=True)
dnnmark_aggr.drop(["tmp"], axis=1, inplace=True)
print("Aggregated df algos")
print(dnnmark_aggr.head(2))

fig, ax = plt.subplots(figsize=(10, 6))
if args.singleVGG:
    dnnmark_aggr[[
        "batch", "VGG time"
    ]].plot("batch", "VGG time", lw=1, alpha=1, ms=4, marker="o", fillstyle="full", markerfacecolor="#ffffff", ax=ax)
    if "max" in clean_logs.columns.values:
        ax.fill_between(dnnmark_aggr["batch"], dnnmark_aggr["VGG max"], dnnmark_aggr["VGG min"], color="red", alpha=0.3)

else:
    dnnmark_aggr[[
        "batch", "VGG time", "algos"
    ]].plot("batch", "VGG time", lw=1, alpha=1, ms=4, marker="o", fillstyle="full", markerfacecolor="#ffffff", ax=ax)

no_errors = True
if args.ref != "":
    try:
        ref_df = pd.read_csv(args.ref)
        if ref_df.shape[0] < 1:
            print("No data found in", args.ref)
            no_errors = False
    except:
        print("Could not read from", args.ref)
        no_errors = False
    if no_errors:
        ref_df.plot(
            "batch",
            "time",
            lw=1,
            ls=":",
            color="grey",
            alpha=1,
            ms=4,
            marker="o",
            fillstyle="full",
            markerfacecolor="#ffffff",
            label="Reference time",
            ax=ax
        )

ax.set_title(dnnmark_aggr.iloc[0]["machine"])
ax.legend()
drawGrid(ax, xstep=50)
plt.subplots_adjust(top=0.85)
fig.suptitle("VGG-aggregated DNNMark time (s)", fontsize=16)

fig.text(
    1.01, 1., meta1 + "\n" + meta2, ha="left", va="top", transform=ax.transAxes, fontsize=7, fontfamily="monospace"
)

text_ = ""
if os.path.exists(os.path.join(logdir, "README")):
    with io.open(os.path.join(logdir, "README"), mode="r", encoding='utf-8') as f:
        text_ = f.read().replace("\\n", "\n")
    text_ += "\n"

text_ += " ".join([str(b) for b in dnnmark_aggr["batch"].unique()]) + "\n"
text_ += logdir + "\n"
if args.ref != "":
    text_ += "Reference time from " + args.ref + "\n"
fig.text(0, 0, text_, ha="left", va="top", fontsize=8)

if args.text:
    text_ = "{:s}".format(args.text.replace(r'\n', "\n"))
    fig.text(0.6, 0, text_, ha="left", va="top", fontsize=8)

fig_file = os.path.join(logdir, "VGG_time_{}.pdf".format(machine))
plt.savefig(fig_file, bbox_inches="tight")
print("Saved plot to", fig_file)

fig_file = os.path.join(logdir, "VGG_time_{}.png".format(machine))
plt.savefig(fig_file, bbox_inches="tight", dpi=144)
print("Saved plot to", fig_file)
plt.close()

# Save VGG time
print("VGG-aggregated data")
csv_file = os.path.join(logdir, "VGG_time_{}.csv".format(machine))
dnnmark_aggr.to_csv(csv_file, index=False)
print(("CSV saved to {}".format(csv_file)))
