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

# Set GPU range
gpus = [0]

# Change hostname
host = multigpuexec.getHostname()  # "mouse.cont"

default_benchmark = "test_one_layer"
benchmark = default_benchmark
default_template = "fc_config"
template = default_template

parser = argparse.ArgumentParser()
parser.add_argument("--text", "-t", default="", help="Notes to save to README file.")
parser.add_argument("--host", default=None, help="Host name")
parser.add_argument("--dir", "-d", default=None, help="Path to logs directory.")
parser.add_argument("--runs",
                    "-r",
                    type=int,
                    default=1,
                    help="Number of runs for each configuration and mini-batch size")
parser.add_argument(
    "--profileruns",
    type=int,
    default=0,
    help="Number of profiling runs for each configuration and mini-batch size")
parser.add_argument('--date', default=None, help='Set date for the logs path.')
parser.add_argument("--template",
                    default=default_template,
                    help="Configuration file template with .dnntemplate extension.")
parser.add_argument("--benchmark",
                    default=default_benchmark,
                    help="Benchmark to use the configuration file with.")
parser.add_argument("--debug",
                    action="store_true",
                    default=False,
                    help="Run DNNMark with --debuginfo option.")
parser.add_argument("--mbs",
                    type=int,
                    default=None,
                    nargs='*',
                    help="Space-separated list of mini-batch sizes.")
args = parser.parse_args()

if args.host:
    host = args.host

if args.template:
    template = args.template

if args.benchmark:
    benchmark = args.benchmark

# Set number of runs
runs = args.runs
profile_runs = args.profileruns

# Set mini-batch sizes
batchsizes = None
if args.mbs is None:
    batchsizes = [5, 6, 7, 8, 9, 10, 12, 15] + list(range(20, 501, 10))
    # batchsizes = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50]
else:
    batchsizes = args.mbs
print("Batches {}".format(batchsizes))

# VGG model convolution shapes
configs = [(25088, 4096), (4096, 4096), (4096, 1000)]

# Use today's date or change to existing logs directory name
date = datetime.datetime.today().strftime('%Y%m%d')
if args.date:
    date = args.date
    print("Using date {}".format(date))

with_memory = False
debuginfo_option = ""
if args.debug:
    debuginfo_option = " --debug"
tasks = []

# Remove for only 1 iteration
datasetsize = 50000

if benchmark != default_benchmark:
    command = "./run_dnnmark_template.sh -b {}".format(benchmark)
else:
    command = "./run_dnnmark_template.sh"

if "/" in benchmark:
    benchmark = benchmark.split("/")[-1]

command += " --template {}".format(template)

logroot = "/host/DNNMark/logs/"
if args.dir:
    logroot = args.dir
logdir = os.path.join(logroot,
                      "{}/dnnmark_{}_microseries_{}/".format(host, benchmark, date))

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)

# Save --text argument to README
if args.text:
    filename = os.path.join(logdir, "README")
    with open(filename, "w") as f:
        f.write(args.text)
        print("Saved {}".format(filename))

logfile_base = "dnnmark_{}_{}".format(host, template)

for run in range(runs):
    for batch in batchsizes:
        if datasetsize > 0:
            iterations = int(math.ceil(datasetsize / batch))
        else:
            iterations = 1
            # print("BS: {}, Iterations: {}".format(batch, iterations))
        for config in configs:
            C, K = config
            logname = "{}_fc{}x{}_bs{}".format(logfile_base, C, K, batch)

            logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
            if os.path.isfile(logfile):
                print("file", logfile, "exists.")
            else:
                command_pars = command + " -c {} -k {} -w 1 -h 1 -n {} --iter {} {}".format(
                    C, K, batch, iterations, debuginfo_option)
                task = {
                    "comm": command_pars,
                    "logfile": logfile,
                    "batch": batch,
                    "nvsmi": with_memory
                }
                tasks.append(task)
for run in range(profile_runs):
    for batch in batchsizes:
        if datasetsize > 0:
            iterations = int(math.ceil(datasetsize / batch))
        else:
            iterations = 10
            # print("BS: {}, Iterations: {}".format(batch, iterations))
        for config in configs:
            C, K = config
            logname = "{}_fc{}x{}_bs{}".format(logfile_base, C, K, batch)
            nvlogname = "{}_iter{}_{:02d}".format(logname, iterations, run)
            logfile = os.path.join(logdir, "{}_%p.nvprof".format(nvlogname))
            if os.path.isfile(logfile):
                print("file", logfile, "exists.")
            else:
                command_pars = command + " -c {} -k {} -w 1 -h 1 -n {} --iter {} {}".format(
                    C, K, batch, iterations, debuginfo_option)
                profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --profile-from-start off --csv --log-file {} {}".format(
                    logfile, command_pars)
                task = {
                    "comm": profcommand,
                    "logfile": logfile,
                    "batch": batch,
                    "conv": conv,
                    "nvsmi": False
                }
                tasks.append(task)

print("Have", len(tasks), "tasks")
gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus,
                                      start=gpu + 1,
                                      c=4,
                                      d=1,
                                      nvsmi=tasks[i]["nvsmi"],
                                      mode="dmon",
                                      debug=False)
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
