#!/usr/bin/env python3

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018-2019 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import datetime
import math
import argparse
import pandas as pd
import sys

# Set GPU range
gpus = [0]

# Change hostname
host = multigpuexec.getHostname()  # "mouse.cont"

template = "conv_alone_config"

parser = argparse.ArgumentParser()
parser.add_argument("--text", "-t", default="", help="Notes to save to README file.")
parser.add_argument("--host", default=None, help="Host name")
parser.add_argument("--dir", "-d", default=None, help="Path to logs directory.")
parser.add_argument("--runs", "-r", type=int, default=1,
                    help="Number of runs for each configuration and mini-batch size")
parser.add_argument(
    "--profileruns", type=int, default=0,
    help="Number of profiling runs for each configuration and mini-batch size")
parser.add_argument('--date', default=None, help='Set date for the logs path.')
parser.add_argument(
    "--template", default=template,
    help="Configuration file template with .dnntemplate extension. Default: {}".format(
        template))
parser.add_argument("--debug", action="store_true", default=False,
                    help="Run DNNMark with --debuginfo option.")
parser.add_argument("--warmup", action="store_true",
                    help="Run warmup before measuring time.")
parser.add_argument(
    "--convconfigfile", default=None,
    help="Model convolutional layers configuration. By default VGG16 cofiguration is used."
)
args = parser.parse_args()

if args.host:
    host = args.host

if args.template:
    template = args.template

# Use today's date or change to existing logs directory name
date = datetime.datetime.today().strftime('%Y%m%d')
if args.date:
    date = args.date
    print("Using date {}".format(date))

logroot = "/host/DNNMark/logs/"
if args.dir:
    logroot = args.dir
logdir = os.path.join(logroot, "{}/dnnmark_ConvSeries_{}/".format(host, date))

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)

# Save --text argument to README
if args.text:
    filename = os.path.join(logdir, "README")
    with open(filename, "w") as f:
        f.write(args.text)
        print("Saved {}".format(filename))

with_memory = False
debuginfo_option = ""
if args.debug:
    debuginfo_option = " --debug"

if args.warmup:
    command += " --warmup 10"

# Remove for only 1 iteration
datasetsize = 50000

# Set number of runs
runs = args.runs
profile_runs = args.profileruns

# Set mini-batch sizes
batchsizes = [150, 500]
# batchsizes = [5, 6, 7, 8, 9, 10, 12, 15] + list(range(20, 501, 10))

# VGG model convolution shapes
if args.convconfigfile is None:
    # Use VGG16 configuraion
    print("Simulate VGG16 convolutional layers.")
    configs = pd.read_csv('dnnmarkConvConfigs/vgg16.csv')
    # [(2, 512, 512), (4, 512, 512), (4, 256, 512), (8, 256, 256), (8, 128, 256),
    #            (16, 128, 128), (16, 64, 128), (32, 64, 64), (32, 3, 64)]
else:
    configs = pd.read_csv(args.convconfigfile)

# Drop duplicate layer configurations
configs.drop_duplicates(inplace=True)
# Strip whitespaces from column names
configs.rename(columns=lambda x: x.strip(), inplace=True)

# Set algorithm combinations
algo_configs = {
    # "all-workspace10MB":
    # "chainer":
    # "--algofwd cudnn --algo cudnn --algod cudnn --fwd_pref specify_workspace_limit --bwd_filter_pref specify_workspace_limit --bwd_data_pref specify_workspace_limit --workspace 10000000"
    # "tf": "--algofwd auto --algo auto --algod auto"
    "pytorch": "--algofwd cudnnv7 --algo cudnnv7 --algod cudnnv7"
}

benchmarks = ['test_fwd_conv', 'test_bwd_conv']

tasks = []

for run in range(runs):
    for algoconf in algo_configs:
        other_options = algo_configs[algoconf]
        for batch in batchsizes:
            for benchmark in benchmarks:
                logfile_base = "dnnmark_{}_{}_{}".format(host, template.replace('_', ''),
                                                         benchmark.replace('_', ''))
                command = "./run_dnnmark_template.sh  --template {} -b {}".format(
                    template, benchmark)
                if datasetsize > 0:
                    iterations = int(math.ceil(datasetsize / batch))
                else:
                    iterations = 1
                    # print("BS: {}, Iterations: {}".format(batch, iterations))
                for config in configs.iterrows():
                    config = config[1]  # through away index
                    H = config['image height']
                    W = config['image width']
                    C = config['input channels']
                    K = config['output channels']
                    S = config['kernel size']
                    P = config['padding']
                    U = config['stride']
                    # imsize, channels, conv = config
                    # print("FWD {}, BWD data {}, BWD filter {}".format(algofwd, algod, algo))
                    logname = "{}_shape{H}-{W}-{C}-{K}-{S}-{P}-{U}_bs{batch}_algos{algos}".format(
                        logfile_base, H=H, W=W, C=C, K=K, S=S, P=P, U=U, batch=batch,
                        algos=algoconf)

                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        command_pars = command + " -h {H} -w {W} -c {C} -k {K} -s {S} -u {U} -n {batch} --iter {iter} {other} {debug}".format(
                            H=H, W=W, C=C, K=K, S=S, P=P, U=U, batch=batch,
                            iter=iterations, other=other_options, debug=debuginfo_option)
                        task = {
                            "comm": command_pars,
                            "logfile": logfile,
                            "batch": batch,
                            "nvsmi": with_memory
                        }
                        tasks.append(task)
for run in range(profile_runs):
    for algoconf in algo_configs:
        other_options = algo_configs[algoconf]
        for batch in batchsizes:
            for benchmark in benchmarks:
                logfile_base = "dnnmark_{}_{}_{}".format(host, template.replace('_', ''),
                                                         benchmark.replace('_', ''))
                command = "./run_dnnmark_template.sh  --template {} -b {}".format(
                    template, benchmark)
                for config in configs.iterrows():
                    config = config[1]  # through away index
                    H = config['image height']
                    W = config['image width']
                    C = config['input channels']
                    K = config['output channels']
                    S = config['kernel size']
                    P = config['padding']
                    U = config['stride']
                    iterations = 10
                    logname = "{}_shape{H}-{W}-{C}-{K}-{S}-{P}-{U}_bs{batch}_algos{algos}".format(
                        logfile_base, H=H, W=W, C=C, K=K, S=S, P=P, U=U, batch=batch,
                        algos=algoconf)
                    nvlogname = "{}_iter{}_{:02d}".format(logname, iterations, run)
                    logfile = os.path.join(logdir, "{}_%p.nvprof".format(nvlogname))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        command_pars = command + " -h {H} -w {W} -c {C} -k {K} -s {S} -u {U} -n {batch} --iter {iter} {other} --warmup 0".format(
                            H=H, W=W, C=C, K=K, S=S, P=P, U=U, batch=batch,
                            iter=iterations, other=other_options)
                        profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --profile-from-start off --csv --log-file {} {}".format(
                            logfile, command_pars)
                        task = {
                            "comm": profcommand,
                            "logfile": logfile,
                            "batch": batch,
                            "nvsmi": False
                        }
                        tasks.append(task)

print("Have", len(tasks), "tasks")
gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=4, d=1,
                                      nvsmi=tasks[i]["nvsmi"], mode="dmon", debug=False)
    gpu_info = multigpuexec.getGPUinfo(
        gpu, query="name,memory.total,memory.free,pstate,clocks.mem,clocks.sm")
    cpu_info = multigpuexec.getCPUinfo()
    f = open(tasks[i]["logfile"], "w+")
    f.write("command:{}\n".format(tasks[i]["comm"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.write("{}".format(cpu_info))
    f.close()
    print(time.strftime("[%d,%H:%M:%S]"))
    multigpuexec.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print("log:", tasks[i]["logfile"])
    print("{}/{} tasks".format(i + 1, len(tasks)))
    time.sleep(0)

print("No more tasks to run. Logs are in", logdir)
