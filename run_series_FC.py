#!/usr/bin/env python3

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import datetime
import math
import sys

# Set GPU range
gpus = list(range(0, 1))

# Change hostname
host = "mouse"

# Set number of runs
runs = 1

# Set mini-batch sizes
# batchsizes = [7, 8, 9] + list(range(10, 50, 2)) + list(range(50, 200, 10)) + list(range(200, 501, 50))
batchsizes = [7, 8, 9] + list(range(10, 200, 10)) + list(range(200, 501, 50))
# batchsizes = [7, 10, 100, 200, 300, 500]


# VGG model convolution shapes (Chaine cifar100 sample)
configs = [(512, 512), (512, 100)]

benchmarks = ["test_fwd_fc", "test_bwd_fc"]
template = "fc_config"

# Use today's date or change to existing logs directory name
date = datetime.datetime.today().strftime('%Y%m%d')
# date = "20190710"

nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []
#other_options = " --bwd_filter_pref specify_workspace_limit  --bwd_data_pref specify_workspace_limit  --fwd_pref specify_workspace_limit "
other_options = ""

# Remove for only 1 iteration
datasetsize = 50000

logdir = "logs/{}/dnnmark_FC_microseries_{}/".format(host, date)
if not os.path.exists(logdir):
    os.makedirs(logdir)
print("Logdir", logdir)
for benchmark in benchmarks:
    command = "./run_dnnmark_template.sh{} --template {} -b {}".format(other_options, template, benchmark)

    logfile_base = "dnnmark_{}_{}".format(host, benchmark)
    for batch in batchsizes:
        if datasetsize > 0:
            iterations = int(math.ceil(datasetsize / batch))
        else:
            iterations = 1
            # print("BS: {}, Iterations: {}".format(batch, iterations))
        for config in configs:
            c, k = config
            logname = "{}_b{}_shape{}-{}".format(
                logfile_base, batch, c, k)
            for run in range(runs):
                logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                if os.path.isfile(logfile):
                    print("file", logfile, "exists.")
                else:
                    command_pars = command + \
                        " -c {} -n {} -k {} -w {} -h {} -d {}{}".format(
                            c, batch, k, 1, 1, datasetsize, debuginfo_option)
                    task = {"comm": command_pars, "logfile": logfile,
                            "batch": batch, "nvsmi": with_memory}
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
                            "batch": batch, "nvsmi": False}
                    tasks.append(task)

print("Have", len(tasks), "tasks")
gpu = -1

for i in range(0, len(tasks)):
    gpu = multigpuexec3.getNextFreeGPU(gpus, start=gpu + 1, c=2, d=1, nvsmi=tasks[i]["nvsmi"], mode="dmon", debug=False)
    gpu_info = multigpuexec3.getGPUinfo(gpu)
    cpu_info = multigpuexec3.getCPUinfo()
    f = open(tasks[i]["logfile"], "w+")
    f.write("command:{}\n".format(tasks[i]["comm"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.write("{}".format(cpu_info))
    f.close()
    print(time.strftime("[%d,%H:%M:%S]"))
    multigpuexec3.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print("log:", tasks[i]["logfile"])
    print("{}/{} tasks".format(i + 1, len(tasks)))
    time.sleep(0)

print("No more tasks to run. Logs are in", logdir)
