#!/usr/bin/env python3
# Parse DNNMark time logs from .log files for all CNNs

import re
import sys
import os
import io
import glob
from cycler import cycler
import matplotlib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import argparse
import time
from datetime import timedelta
from functools import partial

print("Numpy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", sns.__version__)

if 'dnnmark' in os.getcwd().lower():
    sys.path.insert(0, '../lib/')
else:
    sys.path.insert(0, 'lib/')

import lib3
import plotter

version = '1.08h'
condensed_font = 'Roboto Condensed'


# Aggregation function to use instead of median
def aggOperationTimes(s, discard_top=5, av='median', debug=False):
    l = len(s)
    # If discarding more elements than there are in s
    # leave only one element
    if discard_top >= l:
        discard_top = l - 1
    if debug:
        print(f"Series with {l} elements, discarding top {discard_top}")
        print(s[:5])
    s = s.sort_values(ascending=False)
    s = s.iloc[discard_top:]
    if debug:
        print(f"{s.quantile(0.25)} {s.quantile(0.5)} {s.quantile(0.75)}")
    return s.agg(av)


def plotLayersTime(df, grid=None, title="DNNMark layers time (ms)"):
    colors1 = plotter.getColorList('tab20', 20)
    colors2 = plotter.getColorList("viridis", 12)
    colors3 = plotter.getColorList("YlOrBr", 13)
    colors4 = plotter.getColorList("bone", 20)

    colors = colors1 + colors2[1:-1] + colors3[3:-1] + colors4[5:-1]
    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Stacked area plot
    if 'machine' in df:
        data = df.drop(['machine', 'total', 'iterations'], axis=1)
    else:
        data = df.copy()
    data.set_index('batch', drop=True, inplace=True)
    X = data.index.values
    Y = np.transpose(data.values)
    labels = data.columns.get_level_values(0)
    stacks = ax.stackplot(X, Y, labels=labels, lw=0)
    # Line plot - total time
    if 'total' in df.columns:
        df.plot("batch", "total", lw=2, alpha=1, ms=4, marker="o", fillstyle="full",
                markerfacecolor="#ffffff", c="black", ax=ax)
    ax.set_ylabel("time (s)")
    ax.set_xlabel("mini-batch size")
    handles, labels = ax.get_legend_handles_labels()
    # print("Have {} layers".format(len(labels)))
    ncol = 1
    if len(labels) > 24:
        ncol = len(labels) // 24
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=10, ncol=ncol)

    plt.setp(ax.get_legend().get_texts(), family=condensed_font)

    # plt.subplots_adjust(top=0.9)
    fig.suptitle(title, fontsize=16, y=0.95)
    # if grid is not None:
    #     ((x_step, x_subticks), (y_step, y_subticks)) = make_tuple(grid)
    #     plotter.drawGrid(ax, xstep=x_step, ystep=y_step, minor_ticks_x=x_subticks,
    #                      minor_ticks_y=y_subticks)
    # else:
    #     ax.grid(ls=':', lw=0.5, alpha=0.8, which='major')
    #     plt.minorticks_on()
    #     ax.grid(ls=':', lw=0.5, alpha=0.3, which='minor')
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=10))
    ax.grid(ls=':', lw=0.7)
    ax.grid(ls=':', lw=0.3, which='minor')
    ax.set_xlim(0, None)
    return ax


# Plot a FacetGrid of the epoch time per MBS for each convolutional layer
def timePerShapePlot(mlogs, args, metaInfo, error_logs, machine, mbslimits=None, grid=None, title=None):
    logdir = args.dir
    x_step = None
    # if grid is not None:
    #     ((x_step, x_subticks), (y_step, y_subticks)) = make_tuple(args.grid)
    # print("Grid: {}/{}, {}/{}".format(x_step, x_subticks, y_step, y_subticks))

    print("In timePerShapePlot")
    print(mlogs.head())
    pprint(mlogs.sort_values(by=["shape", "mini-batch size"]).head())
    # Limit MBS range to plot
    if mbslimits is not None:
        mlogs = mlogs[(mlogs['mini-batch size'] > mbslimits[0])
                      & (mlogs['mini-batch size'] <= mbslimits[1])]

    if args.CNNperalgo:
        fg = sns.FacetGrid(mlogs.sort_values(by=["shape", "mini-batch size"]), col="shape", hue="algos",
                           height=3, aspect=1.5, margin_titles=True, legend_out=False, sharey=False)
    else:
        fg = sns.FacetGrid(mlogs.sort_values(by=["shape", "mini-batch size"]), col="shape", col_wrap=4,
                           height=3, aspect=1.5, margin_titles=True, legend_out=False, sharey=False)

    if 'itertime max' in mlogs.columns:
        # Fill between min and max
        g = fg.map(plt.fill_between, "mini-batch size", "itertime max", "itertime min", color="red",
                   alpha=0.3, lw=0, label='min-max')
    g = fg.map(plt.plot, "mini-batch size", "itertime", lw=1, alpha=1, ms=4, color='black', marker="o",
               mew=0.5, fillstyle="full", markerfacecolor="#ffffff",
               label="iteration time (s)").add_legend()

    for ax_arr in np.nditer(fg.axes, flags=["refs_ok"]):
        ax = ax_arr.item()
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=10))
        ax.grid(ls=':', lw=0.7)
        ax.grid(ls=':', lw=0.3, which='minor')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    for (i, j, k), data in fg.facet_data():
        if j == 0:
            ax = fg.facet_axis(i, j)
            ax.set_ylabel("iteration time (s)")

    [plt.setp(axg.texts, text="") for axg in g.axes.flat]
    g.set_titles(row_template='{row_name}', col_template='{col_name}', size=9)

    cwd = os.getcwd()
    logdir_path = os.path.join(cwd, logdir)
    split_at = logdir_path.find('logs/')
    logdir_path = logdir_path[split_at:]

    # Text
    text = ''
    readme_path = os.path.join(logdir, "README")
    if os.path.exists(readme_path):
        with io.open(readme_path, mode="r", encoding='utf-8') as f:
            text = f.read()
    text += '\n' + logdir_path + '\n'
    if args.text:
        text += "{:s}".format(args.text.replace(r'\n', "\n")) + '\n'
    # mini-batch sizes
    mbs = [str(a) for a in sorted(mlogs["mini-batch size"].unique())]
    splitsize = 20
    if len(mbs) > splitsize:
        # Split list into groups of splitsize numbers, separate with newlines
        splitmbs = mbs[:splitsize]
        mbs = mbs[splitsize:]
        while len(mbs) > splitsize:
            splitmbs = splitmbs + ['\n'] + mbs[:splitsize]
            mbs = mbs[splitsize:]
        mbs = splitmbs + ['\n'] + mbs
    batchsizes = " ".join(mbs)
    text += batchsizes
    g.fig.text(0, 0, text, ha="left", va="top", fontsize=10, fontfamily=condensed_font)

    # print Meta and errors
    mfont = {'family': 'monospace', 'size': 8}
    text_ = metaInfo.to_string()
    g.fig.text(1, 0, text_, ha="right", va="top", **mfont)
    if error_logs.shape[0] > 0:
        error_logs_ = error_logs[["shape", "batch"]].sort_values(by=["shape", "batch"]).drop_duplicates()
        text_ = "Errors\n" + error_logs_.to_string()
        g.fig.text(0.4, 0, text_, ha="left", va="top", **mfont)

    # plt.subplots_adjust(top=top)
    if title is None:
        plt.suptitle("DNNMark epoch time (s) on {}".format(machine), fontsize=18, va="bottom", y=1.01)
    else:
        plt.suptitle(title, fontsize=18, va="bottom", y=1.01)
    plt.tight_layout()
    fig_file = None
    if mbslimits is None:
        fig_file = os.path.join(logdir, "DNNMark_shape_times.pdf")
    else:
        fig_file = os.path.join(logdir,
                                "DNNMark_shape_times_mbs{}-{}.pdf".format(mbslimits[0], mbslimits[1]))
    plt.savefig(fig_file, bbox_inches="tight")
    print("Saved plot to", fig_file)
    plt.close()


def main(args):
    print('Parse DNNMark logs into CSV files and plot iteration and epoch times v{}'.format(version))

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", default="", help="Notes to place on the plot")
    parser.add_argument("--dir", "-d", default=".", help="Directory with log files")
    parser.add_argument(
        "--av", default='median',
        help="Function for averaging time between runs: median(default)/mean/min/max/25percentile.")
    parser.add_argument(
        "--droptop", type=int, default=1, help=
        "For iteration time per convolution iteration (produced with DNNMark --detailedtime option)" \
        " drop top N elements"
    )
    parser.add_argument("--datasetsize", default=50000, type=int,
                        help="Size of the training dataset. Defines number of iterations per one epoch.")
    parser.add_argument('--reread', action='store_true', default=False,
                        help='Reread log files even if raw_logs.csv exists.')
    parser.add_argument('--reparse', action='store_true', default=False,
                        help='Reparse logs ustin raw_logs.csv data.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Output more information while reading log files.')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug information')
    parser.add_argument('--cnns', type=str, nargs='*', default='all', help="Parse only these CNN times")
    args = parser.parse_args(args)

    configs = "convconfigs/pytorch_imagenet_configs.csv"
    multiline_columns = ["batch", "operation", "gpu_time", "cpu_time"]
    raw_file = os.path.join(args.dir, 'raw_logs.csv')

    start = time.perf_counter()
    if os.path.exists(raw_file) and not args.reread:
        # ---------- Reading form CSV file -----------
        print('Reading logs from CSV file {}'.format(raw_file))
        dflogs = pd.read_csv(raw_file, dtype={"NVdrv": str, "CUDA": str, "OS release": str})
        print(dflogs.head(2))
    else:
        # Mark: Read logs
        output_patterns = [
            re.compile(r"NVDRV:([0-9\.\-]+),CUDA:([0-9\.\-x]+),cuDNN:([0-9\-\.]+)"),
            re.compile(
                r"GPU([0-9]+): ([^,]+), ([0-9]+) MiB, ([0-9]+) MiB, P([0-9]), ([0-9]+), ([0-9]+) MHz, ([0-9]+) MHz, ([0-9a-z]+)"
            ),
            re.compile(r"CPU\(s\):\s+(\d+)"),
            re.compile(r"Model name:\s+(.+)"),
            re.compile(r"CPU MHz:\s+([0-9\.]+)"),
            re.compile(r"CPU max MHz:\s+([0-9\.]+)"),
            re.compile(r"OS kernel:\s*([0-9a-z\-\._]+)"),
            re.compile(r"Release:\s*([0-9\.]+)"),
            re.compile(r"DNNMark suites version ([a-zA-Z0-9\._]+)"),
            re.compile(r"fastiterations:\s*([A-Za-z]+)"),
            re.compile(r"warmup:\s*([0-9\.]+)"),
            re.compile(r"maxtemp:\s*([0-9\.]+)"),
            re.compile(r"taskgroup:\s*([0-9\.]+)"),
            re.compile(r"^c=([0-9]+)"),
            re.compile(r"^h=([0-9]+)"),
            re.compile(r"^w=([0-9]+)"),
            re.compile(r"^conv_mode=([a-zA-Z\-]+)"),
            re.compile(r"^num_output=([0-9]+)"),
            re.compile(r"^kernel_size=([0-9]+)"),
            re.compile(r"^pad=([0-9]+)"),
            re.compile(r"^stride=([0-9]+)"),
            re.compile(r"Benchmark: ([a-zA-Z0-9_]+)"),
            re.compile(r"Iterations:([0-9]+)"),
            re.compile(r"command:.+ --warmup ([0-9]+)"),
            re.compile(
                r"^n=([0-9\.e\+]+) Operation ([a-zA-Z0-9_]+): GPU time ([0-9\.e\+]+)ms, CPU time ([0-9\.e\+]+)ms"
            )
        ]

        column_names = [
            # "time",  #["batch", "FWD time"], ["batch", "BWD time"],
            ["NVdrv", "CUDA", "cuDNN"],
            [
                "GPU_N", "GPU model", "GPU memory.total", "GPU memory.free", "GPU performance",
                "GPU temp", "GPU MEM MHz", "GPU SM MHz, GPU throttle"
            ],
            "CPUs",
            "CPU model",
            "CPU MHz",
            "CPU MHz max",
            "OS kernel",
            "OS release",
            "DNNMark version",
            "fast iterations",
            "warmup",
            "max GPU temp",
            "taskgroup",
            "input channels",
            "image height",
            "image width",
            "conv_mode",
            "output channels",
            "kernel size",
            "padding",
            "stride",
            "benchmark",
            "iterations",
            "warmup (command)",
            multiline_columns
        ]


        filename_patterns_list = [
            re.compile(
                r"^dnnmark_([a-zA-Z0-9@\.]+)_([a-z]+)_test[fb]wdconvmultibs_" \
                "shape([0-9\-]+)_algos([a-zA-Z0-9\-]+)_([0-9]+)\.log$"
            ),
        ]
        columns_lists = [
            ["machine", "config", "shape", "algos", "run"],
        ]

        print("Reading from", args.dir)

        # Loop through different filename patterns
        for filename_pattern, columns in zip(filename_patterns_list, columns_lists):
            pars = {
                "output_patterns": output_patterns,
                "parameters": column_names,
                "filename_pattern": filename_pattern,
                "columns": columns
            }

            dflogs = lib3.readLogs(args.dir, pars, list_values_in_cells=True, parse_column_names=None,
                                   verbose=args.verbose, debug=args.debug, maxfiles=None)
            # Finish loop if read some logs
            if dflogs is not None and dflogs.shape[0] > 0:
                break
        print(f"Read {dflogs.shape[0]} rows")
        dflogs = dflogs.dropna(axis=1, how='all')
        print("Converting column types to numeric")
        int_columns = [
            c for c in [
                "batch", "run", "iterations", "warmup", "GPU_N", "CPUs", "input channels",
                "image height", "image width", "output channels", "kernel size", "padding", "stride"
            ] if c in dflogs.columns
        ]

        for col in int_columns:
            dflogs[col] = pd.to_numeric(dflogs[col], errors='coerce', downcast='integer')

        # dflogs[int_columns] = dflogs[int_columns].fillna(-1).astype(int).replace(-1, np.nan)
        print("dflogs int columns")
        print(dflogs[int_columns].head(2))
        time_cols = [
            t for t in dflogs.columns if 'time' in t.lower() or 'bwd' in t.lower() or 'fwd' in t.lower()
        ]
        for time_column in time_cols:
            dflogs[[time_column]] = dflogs[[time_column]].astype(float)
            # Convert ms to s
            dflogs[time_column] = dflogs[time_column] / 1000.
        print("dflogs multiline_columns")
        print(dflogs[multiline_columns].head())
        dflogs = dflogs.sort_values(by=["batch"])
        # Use shape from filenames, do not execute the following lines
        # print("Generating shape column")
        # # Conv.layer shape
        # dflogs.loc[:, 'parsed_shape'] = dflogs.apply(lib3.shapeFromRow, axis=1)
        # print(dflogs[['shape', 'parsed_shape']].drop_duplicates().head())
        raw_file = os.path.join(args.dir, 'raw_logs.csv')
        dflogs.to_csv(raw_file, index=False)
        print("Saved RAW logs into", raw_file)
    end_reading = time.perf_counter()
    print("Logs loaded into memory in {}.".format(timedelta(seconds=(end_reading - start))))
    # ---------- End reading log files ---------------

    print(dflogs.head())
    benchmarks = dflogs['benchmark'].unique()
    print("Benchmarks {}".format(benchmarks))
    # Deal with nans in benchmark column
    if None in benchmarks or np.nan in benchmarks:
        print("None in benchmarks. Dropping the following rows:")
        print(dflogs[dflogs['benchmark'].isna()])
        dflogs = dflogs[dflogs['benchmark'].notna()]
        benchmarks = dflogs['benchmark'].unique()
    fwd_bwd_logs = True  # Separate log files for FWD and BWD convolutions
    for benchmark in benchmarks:
        if '_fwd_' not in benchmark and '_bwd_' not in benchmark:
            fwd_bwd_logs = False
            break
    print("Have FWD/BWD benchmarks?", fwd_bwd_logs)
    keep_columns = list(set(['shape', 'batch', 'run', 'iterations']) | set(multiline_columns))
    if fwd_bwd_logs:
        # keep_columns += ['FWD time', 'BWD time']
        fwdbwd_columns = [c for c in dflogs.columns if 'fwd' in c.lower() or 'bwd' in c.lower()]
        keep_columns += fwdbwd_columns
    else:
        keep_columns += ['time']
    meta_columns = list(set(dflogs.columns) - set(keep_columns) - set(lib3.shape_parameters))
    metaInfo = pd.Series(lib3.aggregateMeta(dflogs[meta_columns]))
    metaInfo['iterations'] = lib3.describeValuesOf(dflogs['iterations'])
    runs = len(dflogs["run"].unique())
    metaInfo['runs'] = runs
    metaInfo = metaInfo.sort_index(key=lambda x: x.str.lower())
    print("Columns in metaIfno")
    print(','.join(metaInfo.index))
    dflogs = dflogs[keep_columns].sort_values(['shape', 'batch'])
    redundant_columns = ['iterations', 'time', 'FWD time', 'BWD time']
    dflogs = dflogs.drop(redundant_columns, axis=1, errors='ignore')

    itertime_columns = multiline_columns
    itertime_columns.remove('operation')
    print("itertime columns:", itertime_columns)
    # Convert itertime_columns to float
    for col in itertime_columns:
        dflogs[col] = pd.to_numeric(dflogs[col], errors='coerce', downcast='float')
    intcolumns = ['batch', 'run']
    for col in intcolumns:
        dflogs[col] = pd.to_numeric(dflogs[col], errors='coerce', downcast='integer')

    print("DFlogs before aggregation")
    print(dflogs.head())

    # Aggregate time of each operation (multiple iteration times)
    # Aggregate time of multiple iterations for each operation
    aggOperationsPartial = partial(aggOperationTimes, discard_top=args.droptop, av=args.av, debug=False)
    dflogs = dflogs.pivot_table(index=['batch', 'shape', 'run'], columns=['operation'],
                                aggfunc=aggOperationsPartial)
    dflogs = dflogs.reset_index(drop=False)

    print(f"dflogs aggregated operation times using {args.av} and droptop {args.droptop}")
    print(dflogs.head())

    # Mark: Aggregate CNN times
    config_files = glob.glob("convconfigs/pytorch_*_imagenet.csv")
    for config_file in config_files:
        if '_all_' in config_file:
            continue

        cnn = config_file.split('/')[-1].split('.')[0]
        cnn = cnn.replace('pytorch_', '').replace('_imagenet', '')
        if args.cnns != 'all' and cnn not in args.cnns:
            continue
        print(cnn)
        config = pd.read_csv(config_file)
        # Strip whitespaces from column names
        config.rename(columns=lambda x: x.strip(), inplace=True)
        # Need not to count BWD time of the first layer
        # Mark the first layer
        config.loc[:, 'shape'] = config.apply(lib3.shapeFromRow, axis=1)
        firstlayershape = config.loc[0, 'shape']

        # Count layer shapes
        config.loc[:, "count"] = 0
        # Mark the first layer of CNN -  need to ignore BWD pass time
        config.loc[:, 'first'] = False
        config.loc[0, 'first'] = True
        assert config.loc[0, 'shape'] == firstlayershape

        config = config[['shape', 'first', 'count']].groupby(['shape', 'first'],
                                                             as_index=False).agg('count')

        config.columns = pd.MultiIndex.from_product([config.columns, ['']])
        column = ('shape', '')

        cnntime = dflogs.merge(config, on=[column], how='right')
        cnntime = cnntime.sort_values([('batch', ''), ('first', ''), ('shape', ''), ('run', '')],
                                      ascending=[True, False, True, True])
        cnntime.columns = cnntime.columns.set_names('operation', level=1)

        # Check missing layer times
        missingtimes = cnntime[cnntime['batch'].isnull()]
        if missingtimes.shape[0] > 0:
            print(f"Missing layers in {cnn}")
            print(missingtimes['shape'])
            continue

        # Check missing MBS
        print("Checking for missing layers/MBSs")
        print(cnntime.head())
        checkdf = cnntime.copy()
        checkdf = checkdf.pivot_table(index=['batch'], columns=['shape'], values=['gpu_time'])
        error_mbss = sorted(checkdf[checkdf.isna().any(axis=1)].index.values)
        checkdf = checkdf.dropna()
        keepmbs = sorted(checkdf.index.values)
        if len(error_mbss) > 0:
            print("Errors in ", lib3.list2shortstring(error_mbss))
            print("Keep MBS:", lib3.list2shortstring(keepmbs))
        cnntime = cnntime[cnntime['batch'].isin(keepmbs)]

        print("CNNtime")
        print(cnntime.head())
        #           batch               shape run      cpu_time                                gpu_time                           first count
        # operation                               ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0 ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0
        # 120           1  224-224-3-64-7-3-2   0      0.000015        0.000015  0.000013      0.000058        0.000048  0.000039  True     1
        # 122           5  224-224-3-64-7-3-2   0      0.000015        0.000015  0.000011      0.000173        0.000183  0.000116  True     1

        # Do not count BWD Data time for the first layer (first==True)
        # Set Bwd time to zero for the first layer
        for c in cnntime.columns:
            if 'bwddata' in c[1].lower():
                cnntime.loc[cnntime['first'], c] = 0

        # Aggregate operation times to iteration times (CPU and GPU)
        #            cpu_time                                gpu_time
        # operation  ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0 ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0
        cnntime.loc[:, ('cpu_itertime', '')] = cnntime['cpu_time'].agg('sum', axis=1)
        cnntime.loc[:, ('itertime', '')] = cnntime['gpu_time'].agg('sum', axis=1)
        print("Aggregated iteertimes in CNN time")
        print(cnntime.head())
        #          batch                  shape run      cpu_time                                gpu_time                            first count cpu_itertime  itertime
        # operation                                  ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0 ConvBwdData_0 ConvBwdFilter_0 ConvFwd_0
        # 120           1     224-224-3-64-7-3-2   0      0.000000        0.000000  0.000013      0.000000        0.000000  0.000039   True     1     0.000013  0.000039
        # 121           1     224-224-3-64-7-3-2   1      0.000000        0.000000  0.000012      0.000000        0.000000  0.000038   True     1     0.000012  0.000038
        # 0             1  14-14-1024-2048-1-0-2   0      0.000013        0.000023  0.000015      0.000180        0.000046  0.000071  False     1     0.000052  0.000297

        folder = os.path.join(args.dir, cnn)
        if not os.path.exists(folder):
            os.mkdir(folder)
        csv_file = 'raw_time.csv'
        cnntime.to_csv(os.path.join(folder, csv_file), index=False)
        print(f"Saved {os.path.join(folder,csv_file)}")

        # Drop unused columns
        cnntime = cnntime.drop(['cpu_time', 'gpu_time', 'first'], axis=1, level=0)
        cnntime = cnntime.droplevel('operation', axis=1)

        cnntime['itertime'] = cnntime['itertime'] * cnntime['count']
        cnntime['cpu_itertime'] = cnntime['cpu_itertime'] * cnntime['count']
        cnntime = cnntime.drop(['count'], axis=1)
        cnntime['batch'] = cnntime['batch'].astype(int)
        #      batch               shape  run  cpu_itertime  itertime
        # 180    1    224-224-3-64-7-3-2    0      0.000016   0.00005
        # 181    1    224-224-3-64-7-3-2    1      0.000017   0.00005

        # Mark: Plot layer times

        # Aggregate runs
        # Mark: cnntimeagg not used!
        cnntimeagg = cnntime.groupby(['batch', 'shape'], as_index=False).agg(args.av).drop(['run'],
                                                                                           axis=1)

        print("Aggregated runs")
        #    batch                  shape  cpu_itertime  itertime
        # 0      1  14-14-1024-2048-1-0-2      0.000053  0.000233
        # 1      1   14-14-1024-256-1-0-1      0.001768  0.002820

        mbs = sorted(cnntime['batch'].unique())
        mbs_str = lib3.list2shortstring(mbs)
        for itertimecolumn in ['cpu_itertime', 'itertime']:
            short_piv = cnntime.pivot_table(index='batch', columns='shape', values=itertimecolumn)
            short_piv.reset_index(inplace=True)
            ax = plotLayersTime(short_piv, title="DNNMark layers time for {} (ms)".format(cnn))
            text_ = ""
            if os.path.exists(os.path.join(args.dir, "README")):
                with io.open(os.path.join(args.dir, "README"), mode="r", encoding='utf-8') as f:
                    text_ = f.read().replace("\\n", "\n")
                text_ += "\n"
            text_ += args.dir + "\n"
            if args.text:
                text_ += "{:s}".format(args.text.replace(r'\n', "\n"))

            text_ += "MBSs:" + mbs_str + "\n"

            ax.text(0, -0.1, text_, ha="left", va="top", fontsize=7, transform=ax.transAxes,
                    fontfamily='Roboto Condensed')

            fig_file = os.path.join(args.dir, cnn, "{}_per_layer.pdf".format(itertimecolumn))
            plt.savefig(fig_file, bbox_inches="tight", dpi=144)
            print("Saved iteration times plot to", fig_file)

        # Mark: aggregate CNN time (sum time for all shapes)
        cnntime = cnntime.drop(['shape'], axis=1)
        # cnntime = cnntime[['run', 'batch', 'shape', 'itertime', 'cpu itertime']]
        print("Calculated itertime per shape")
        #     batch  run  cpu_itertime  itertime
        # 60      1    0      0.000013  0.000039
        # 61      1    1      0.000012  0.000038

        cnntime = cnntime.groupby(['batch', 'run'], as_index=False).agg('sum')
        cnntime = cnntime.rename(columns={
            'itertime': 'CNN itertime',
            'cpu_itertime': 'CNN cpu itertime'
        })

        # Aggregate runs
        cnntime = cnntime.groupby('batch').agg(['min', args.av, 'max'])
        cnntime = cnntime.drop('run', axis=1).reset_index(drop=False)
        # Flatten columns
        cnntime.columns = [c[0] if c[1] == '' else c[0] + ' ' + c[1] for c in cnntime.columns.values]
        # Rename median columns
        cnntime.columns = [c.replace(' ' + args.av, '') for c in cnntime.columns]
        cnntime.loc[:, 'rel_range'] = (cnntime['CNN itertime max'] -
                                       cnntime['CNN itertime min']) / cnntime['CNN itertime'] * 100.

        # Epoch time
        # GPU time
        for c in cnntime.columns:
            if "CNN itertime" in c:
                epochtime_c = c.replace('itertime', 'time')
                cnntime.loc[:, epochtime_c] = cnntime[c] * (args.datasetsize / cnntime["batch"])
            elif "CNN cpu itertime" in c:
                epochtime_c = c.replace('itertime', 'time')
                cnntime.loc[:, epochtime_c] = cnntime[c] * (args.datasetsize / cnntime["batch"])

        metaInfo['CNN model'] = cnn
        metaInfo['variability'] = cnntime['rel_range'].mean()
        metaInfo['aggregation'] = args.av
        metaInfo['date'] = parseDateFromPath(args.dir)
        mbs = sorted(cnntime['batch'].unique())
        mbs_str = lib3.list2shortstring(mbs)
        metaInfo['mbss'] = mbs_str
        metaInfo['parser'] = version
        metaInfo['droptop'] = args.droptop
        metaInfo = metaInfo.sort_index()

        if not os.path.exists(folder):
            os.mkdir(folder)
        csv_file = 'CNN_time.csv'
        cnntime.to_csv(os.path.join(folder, csv_file), index=False)
        print(f"Saved {os.path.join(folder,csv_file)}")
        plotCNNtimes(cnntime, cnn, config_file, metaInfo, folder, av=args.av, error_mbs=error_mbss)

        # META info
        if os.path.exists(os.path.join(folder, "../", "README")):
            with io.open(os.path.join(folder, "../", "README"), mode="r", encoding='utf-8') as f:
                metaInfo['readme'] = f.read().replace("\\n", "\n")
        if 'benchmark' in metaInfo and 'benchmark_filename' in metaInfo:
            del metaInfo['benchmark_filename']
        metaInfo['time_CSV'] = csv_file
        metaInfo = metaInfo.sort_index(key=lambda x: x.str.lower())
        # Save metainfo
        fname = 'INFO.csv'
        infofile = os.path.join(folder, fname)
        metadf = metaInfo.to_frame()
        metadf = metadf.sort_index()
        metadf.to_csv(infofile, header=None)
        print('Saved meta info into {}.'.format(infofile))

    print("Done.")


# MARK: plot CNN time
def plotCNNtimes(df, cnn, convconfig, metaInfo, folder, av='median', error_mbs=None):
    condensed_font = 'Roboto Condensed'
    # Variability of GPU iteration time as mean of relative ranges
    metaInfo['variability'] = df['rel_range'].mean()
    # Form a string representing the list of used mbs
    mbs = sorted(df['batch'].unique())
    mbs_str = lib3.list2shortstring(mbs)
    # Color palette
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10.colors)
    # Plot CNN-aggregated time
    for cnntime in ["CNN time", "CNN itertime"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        if cnntime + " max" in df.columns.values:
            ax.fill_between(df["batch"], df[cnntime + " max"], df[cnntime + " min"], alpha=0.2,
                            label=cnntime + " min-max")
        df[["batch", cnntime]].plot("batch", cnntime, lw=1, alpha=1, ms=4, marker="o", fillstyle="full",
                                    markerfacecolor="#ffffff", ax=ax, label=cnntime)
        # Plot CPU time
        cnncputime = cnntime.replace("CNN ", "CNN cpu ")
        if cnncputime in df.columns:
            if cnncputime + " max" in df.columns.values:
                ax.fill_between(df["batch"], df[cnncputime + " max"], df[cnncputime + " min"], alpha=0.2,
                                label=cnncputime + " min-max")
            df[["batch",
                cnncputime]].plot("batch", cnncputime, lw=1, alpha=1, ms=4, marker="o", fillstyle="full",
                                  markerfacecolor="#ffffff", ax=ax, label=cnncputime)

        ax.set_title(f"{cnn}")
        #         ax.set_title("{}. Mean variability: {:.2f}%.".format(machine, mean_variability))
        ax.legend()
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.set_ylabel('time (s)')
        ax.set_xlabel('mini-batch size')
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(n=10))
        ax.grid(ls=':', lw=0.7)
        ax.grid(ls=':', lw=0.3, which='minor')
        plt.subplots_adjust(top=0.88)
        fig.suptitle("{} convolutional layers ".format(cnn), fontsize=16)

        fig.text(1.01, 1., metaInfo.to_string(), ha="left", va="top", transform=ax.transAxes, fontsize=7,
                 fontfamily="monospace")

        text_ = ""
        if os.path.exists(os.path.join(folder, "../", "README")):
            with io.open(os.path.join(folder, "../", "README"), mode="r", encoding='utf-8') as f:
                text_ += f.read().replace("\\n", "\n")
            text_ += "\n"

        text_ += "Convolution configurations: {}    Averaging: {}\n".format(convconfig, av)
        text_ += folder + "\n"
        #         if args.ref != "":
        #             text_ += "Reference time from " + args.ref + "\n"
        text_ += "MBSs: " + mbs_str + "\n"
        if error_mbs is not None:
            text_ += "Errors in: " + lib3.list2shortstring(error_mbs) + '\n'

        fig.text(0, 0, text_, ha="left", va="top", fontsize=8, fontfamily=condensed_font)
        fig_file = os.path.join(folder, "{}.pdf".format(cnntime.replace(" ", "_")))
        plt.savefig(fig_file, bbox_inches="tight")
        print("Saved plot to", fig_file)
        plt.close()


#Parse date from directory name
def parseDateFromPath(path):
    substr_list = path.split('_')
    for substr in substr_list[::-1]:
        match = re.match(r'\d{8}', substr)
        if match:
            return match.group()
    return "00000000"  #default


if __name__ == '__main__':
    main(sys.argv[1:])
