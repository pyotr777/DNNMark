#!/usr/bin/env python3

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018-2020 (C) Peter Bryzgalov @ CHITECH Stair Lab

import time
import os
import datetime
import math
import argparse
import pandas as pd
import sys

sys.path.append('../lib')
import multigpuexec
import ast

version = '1.15a'
print(
    'Running DNNMark convolutional layer simulations using test_fwd_conv and test_bwd_conv benchmarks.'\
    '\nScript v{}.'.format(version)
)

start_time = time.strftime("[%d %H:%M:%S]")

# Set GPU range
gpus = [0]

# Change hostname
host = multigpuexec.getHostname()  # "mouse.cont"

template = "conv_alone_config"

parser = argparse.ArgumentParser()
parser.add_argument("--text", "-t", default="", help="Notes to save to README file.")
parser.add_argument("--host", default=None, help="Host name")
parser.add_argument("--dir", "-d", default=None,
                    help="Path to the top level of dnnmark logs directories.")
parser.add_argument(
    "--logdir", default=None, help=
    "Path to the log directory. If the case this options is set, no need to provide --dir option."
)
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
parser.add_argument("--policy", default='pytorch',
                    help='Convolution algorithms selection policy.')
parser.add_argument("--debug", action="store_true", default=False,
                    help="Run DNNMark with --debuginfo option.")
parser.add_argument("--warmup", action="store_true",
                    help="Run warmup before measuring time.")
parser.add_argument(
    "--convconfig", default=None,
    help="Model convolutional layers configuration. By default VGG16 cofiguration is used."
)
parser.add_argument("--model", default=None,
                    help="CNN model to simulate with convolutional layers.")
parser.add_argument("--gpus", default=None, help="GPU numbers (as a Python list).")
parser.add_argument("--iter", default=None,
                    help="Use fixed number of iterations per epoch.")
parser.add_argument(
    "--datasetsize", default=50000, type=int,
    help="Size of the training dataset. Defines number of iterations per one epoch.")
parser.add_argument("--benchmarks", default=None,
                    help="DNNMark benchmark(s) (as a Python list).")
parser.add_argument("--mbs", type=int, default=None, nargs='*',
                    help="Space-separated list of mini-batch sizes.")
parser.add_argument("--dnnmark", default="/DNNMark",
                    help="Driectory with a built DNNMark.")
parser.add_argument('--maxusage', type=int, default=20,
                    help="GPU utilization threshold to consider GPU is free.")
parser.add_argument('--maxtemp', type=int, default=65.,
                    help="Maximum GPU temperature. Use for running tasks on cooler GPUs.")
args = parser.parse_args()

# Change hostname
host = multigpuexec.getHostname()
if args.host:
    host = args.host

if args.iter is not None:
    print("Using {} iteration in epoch for each mini-batch size".format(args.iter))

gpus = [0]
if args.gpus is not None:
    gpus = [int(g) for g in ast.literal_eval(args.gpus)]
print("GPUs: {}".format(gpus))

if args.template:
    template = args.template

# Set number of runs
runs = list(range(args.runs))

print("RUNS: {}".format(runs))

profile_runs = args.profileruns

# Set mini-batch sizes
if args.mbs is None:
    batchsizes = [5, 6, 7, 8, 9, 10, 12, 15] + list(range(20, 501, 10))
else:
    batchsizes = args.mbs
print("Mini-batch sizes: {}".format(batchsizes))

# VGG model convolution shapes
model = 'VGG16'  # default
configs_path = "ConvConfigs"
if args.model is not None:
    model = args.model
    configs = pd.read_csv('{}/{}.csv'.format(configs_path, model.lower()))
else:
    if args.convconfig is None:
        # Use VGG16 configuraion
        configs = pd.read_csv('{}/{}.csv'.format(configs_path, model.lower()))
        # [(2, 512, 512), (4, 512, 512), (4, 256, 512), (8, 256, 256), (8, 128, 256),
        #            (16, 128, 128), (16, 64, 128), (32, 64, 64), (32, 3, 64)]
    else:
        model = '.'.join(os.path.basename(args.convconfig).split('.')[:-1])
        configs = pd.read_csv(args.convconfig)

# Drop duplicate layer configurations
configs.drop_duplicates(inplace=True)
# Strip whitespaces from column names
configs.rename(columns=lambda x: x.strip(), inplace=True)

# Set algorithm combinations
algo_configs = {
    "tf": "--algofwd auto --algo auto --algod auto",
    "pytorch": "--algofwd cudnnv7 --algo cudnnv7 --algod cudnnv7"
}
algoconf = algo_configs[args.policy]
multigpuexec.message("Simulating convolutional layers of {} for {}.".format(
    model, args.policy))
print('Model architecture: {}.'.format(model))
# Use today's date or change to existing logs directory name
date = datetime.datetime.today().strftime('%Y%m%d')
if args.date:
    date = args.date
    print("Using date {}".format(date))

with_memory = False
debuginfo_option = ""
if args.debug:
    debuginfo_option = " --debug"

# Remove for only 1 iteration
datasetsize = 50000
if args.datasetsize is not None:
    datasetsize = args.datasetsize

if args.dnnmark is not None:
    command = os.path.join(args.dnnmark, "run_dnnmark_template.sh")
    command += " --root {}".format(args.dnnmark)
else:
    command = "./run_dnnmark_template.sh"

command += " --template {}".format(template)
if args.warmup:
    command += " --warmup 10"

logdir = None
if args.logdir is None:
    logroot = "/host/DNNMark/logs/"
    if args.dir:
        logroot = args.dir
    logdir = os.path.join(
        logroot, "{}/dnnmark_{}_{}_ConvSeries_{}/".format(host, args.policy, model, date))
else:
    logdir = args.logdir

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)

# Save --text argument to README
if args.text:
    filename = os.path.join(logdir, "README")
    with open(filename, "w") as f:
        f.write(args.text)
        print("Saved {}".format(filename))

benchmarks = ['test_fwd_conv', 'test_bwd_conv']
if args.benchmarks is not None:
    benchmarks = ast.literal_eval(args.benchmarks)
print("Using benchmark(s) {}".format(' '.join(benchmarks)))

tasks = []
for run in runs:
    other_options = algoconf
    for batch in batchsizes:
        for benchmark in benchmarks:
            logfile_base = "dnnmark_{}_{}_{}".format(host, template.replace('_', ''),
                                                     benchmark.replace('_', ''))
            if args.iter is not None:
                iterations = int(args.iter)
                last_mbs = batch
            elif datasetsize > 0:
                iterations = int(math.ceil(datasetsize / batch))
                last_mbs = datasetsize - (iterations - 1) * batch
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
                    algos=args.policy)
                if last_mbs == batch:
                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        command_pars = command + " -b {benchmark} -h {H} -w {W} -c {C} -k {K} -s {S} -p {P} -u {U} -n {batch} --iter {iter} {other} {debug}".format(
                            benchmark=benchmark, H=H, W=W, C=C, K=K, S=S, P=P, U=U,
                            batch=batch, iter=iterations, other=other_options,
                            debug=debuginfo_option)
                        task = {
                            "comm": command_pars,
                            "logfile": logfile,
                            "nvsmi": with_memory
                        }
                        tasks.append(task)
                else:
                    # Last iteration may have smaller mbs if datasetsize is not dividable by the desired mbs.
                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        # Run iterations - 1 with mbs = batch
                        command_pars = command + " -b {benchmark} -h {H} -w {W} -c {C} -k {K} -s {S} -p {P} -u {U} -n {batch} --iter {iter} {other} {debug}".format(
                            benchmark=benchmark, H=H, W=W, C=C, K=K, S=S, P=P, U=U,
                            batch=batch, iter=iterations - 1, other=other_options,
                            debug=debuginfo_option)
                        task = {
                            "comm": command_pars,
                            "logfile": logfile,
                            "nvsmi": with_memory
                        }
                        tasks.append(task)

                    logfile = os.path.join(logdir, "{}_{:02d}a.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        # Run 1 iteration with mbs = last_mbs
                        command_pars = command + " -b {benchmark} -h {H} -w {W} -c {C} -k {K} -s {S} -p {P} -u {U} -n {batch} --iter {iter} {other} {debug}".format(
                            benchmark=benchmark, H=H, W=W, C=C, K=K, S=S, P=P, U=U,
                            batch=last_mbs, iter=1, other=other_options,
                            debug=debuginfo_option)
                        task = {
                            "comm": command_pars,
                            "logfile": logfile,
                            "nvsmi": with_memory
                        }
                        tasks.append(task)

for run in range(profile_runs):
    other_options = algoconf
    for batch in batchsizes:
        for benchmark in benchmarks:
            logfile_base = "dnnmark_{}_{}_{}".format(host, template.replace('_', ''),
                                                     benchmark.replace('_', ''))
            for config in configs.iterrows():
                command_pars = command + " -b {}".format(benchmark)
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
                    command_pars += " -h {H} -w {W} -c {C} -k {K} -s {S} -u {U} -n {batch} --iter {iter} {other} --warmup 0".format(
                        H=H, W=W, C=C, K=K, S=S, P=P, U=U, batch=batch, iter=iterations,
                        other=other_options)
                    profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --profile-from-start off --csv --log-file {} {}".format(
                        logfile, command_pars)
                    task = {"comm": profcommand, "logfile": logfile, "nvsmi": False}
                    tasks.append(task)

print("Have", len(tasks), "tasks")
gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=3, d=1,
                                      nvsmi=tasks[i]["nvsmi"], mode="dmon",
                                      maxusage=args.maxusage, maxtemp=args.maxtemp,
                                      debug=False)
    gpu_info = multigpuexec.getGPUinfo(
        gpu, query="name,memory.total,memory.free,pstate,clocks.mem,clocks.sm")
    cpu_info = multigpuexec.getCPUinfo()
    os_info = multigpuexec.getOSinfo()
    f = open(tasks[i]["logfile"], "w+")
    f.write("command:{}\n".format(tasks[i]["comm"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.write("{}\n".format(cpu_info))
    f.write("{}\n".format(os_info))
    f.close()
    print(time.strftime("[%d %H:%M:%S]"))
    multigpuexec.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print("log:", tasks[i]["logfile"])
    print("{}/{} tasks".format(i + 1, len(tasks)))
    time.sleep(0)

multigpuexec.waitComplete()  # Wait for all running tasks to complete
print("No more tasks to run.\nLogs are in {}".format(logdir))

# Save file-flag that all tasks are finished.
end_time = time.strftime("[%d %H:%M:%S]")
filename = os.path.join(logdir, "DONE")
with open(filename, "w") as f:
    f.write("Done {} tasks.\nTime: {} - {}\n".format(len(tasks), start_time, end_time))
    print("Saved {}".format(filename))
