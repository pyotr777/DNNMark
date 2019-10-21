#!/usr/bin/env python3

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018-2019 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import datetime
import math

# Set GPU range
gpus = list(range(0, 1))

# Change hostname
host = "mouse"

# Set number of runs
runs = 1

# Set mini-batch sizes
batchsizes = [7, 8, 9] + list(range(10, 200, 10)) + list(range(200, 501, 50))
# batchsizes = [7, 10, 20, 30, 100, 190, 200, 300, 500]

# Set algorithm combinations
algo_configs = [
    ["cudnn", "cudnn", "cudnn"]  # ,
    # [None, None, None]
]

# VGG model convolution shapes
configs = [(2, 512, 512), (4, 512, 512), (4, 256, 512), (8, 256, 256),
           (8, 128, 256), (16, 128, 128), (16, 64, 128), (32, 64, 64), (32, 3, 64)]

benchmark = "test_composed_model"
default_benchmark = "test_composed_model"

# Use today's date or change to existing logs directory name
date = datetime.datetime.today().strftime('%Y%m%d')
# date = "20190926"

nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []
other_options_list = ["--bwd_filter_pref fastest --bwd_data_pref fastest  --fwd_pref fastest ",
                      "--bwd_filter_pref fastest  --bwd_data_pref fastest  --fwd_pref specify_workspace_limit --workspace 200000000 ",
                      "--bwd_filter_pref specify_workspace_limit  --bwd_data_pref fastest  --fwd_pref fastest --workspace 200000000 "]
other_options_names = ["all-fastest", "fwd-workspace200MB", "filter-workspace200MB"]
# other_options = ""

# Remove for only 1 iteration
datasetsize = 50000

if benchmark != default_benchmark:
    command = "./run_dnnmark_template.sh -b {}".format(benchmark)
else:
    command = "./run_dnnmark_template.sh"

if "/" in benchmark:
    benchmark = benchmark.split("/")[-1]
logdir = "logs/{}/dnnmark_{}_microseries_{}/".format(host, benchmark, date)

if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)

for other_options, options_name in zip(other_options_list, other_options_names):
    command_options = command + " {}".format(other_options)
    logfile_base = "dnnmark_{}_{}_{}".format(host, benchmark, options_name)
    for algos in algo_configs:
        algofwd, algo, algod = algos
        for batch in batchsizes:
            if datasetsize > 0:
                iterations = int(math.ceil(datasetsize / batch))
            else:
                iterations = 1
                # print("BS: {}, Iterations: {}".format(batch, iterations))
            for config in configs:
                imsize, channels, conv = config
                # print("FWD {}, BWD data {}, BWD filter {}".format(algofwd, algod, algo))
                logname = "{}_shape{}-{}-{}_bs{}_algos{}-{}-{}".format(
                    logfile_base, imsize, channels, conv, batch, algofwd, algo, algod)
                for run in range(runs):
                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        if algo is None:
                            command_pars = command_options + \
                                " -c {} -n {} -k {} -w {} -h {} -d {}{}".format(
                                    channels, batch, conv, imsize, imsize, datasetsize, debuginfo_option)
                        else:
                            command_pars = command_options + " -c {} -n {} -k {} -w {} -h {} --algo {} --algod {} --algofwd {} -d {}{}".format(
                                channels, batch, conv, imsize, imsize, algo, algod, algofwd, datasetsize, debuginfo_option)
                        task = {"comm": command_pars, "logfile": logfile,
                                "batch": batch, "conv": conv, "nvsmi": with_memory}
                        tasks.append(task)
                if nvprof:
                    iterations = 10
                    nvlogname = "{}_iter{}".format(logname, iterations)
                    command_pars = command_options + " -c {} -n {} -k {} -w {} -h {} --algo {} --algod {} --algofwd {} --iter {} --warmup 0".format(
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
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=2, d=1, nvsmi=tasks[i]["nvsmi"], mode="dmon", debug=False)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    cpu_info = multigpuexec.getCPUinfo()
    f = open(tasks[i]["logfile"], "w+")
    f.write("command:{}\n".format(tasks[i]["comm"]))
    f.write("b{} conv{}\n".format(tasks[i]["batch"], tasks[i]["conv"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.write("{}".format(cpu_info))
    f.close()
    print(time.strftime("[%d,%H:%M:%S]"))
    multigpuexec.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print("log:", tasks[i]["logfile"])
    print("{}/{} tasks".format(i + 1, len(tasks)))
    time.sleep(0)

print("No more tasks to run. Logs are in", logdir)
