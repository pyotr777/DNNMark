#!/usr/bin/env python

# Run VGG benchmark series.
# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import datetime

# Set GPU range
gpus = range(0, 1)

# Change hostname
host = "p3.2xlarge"

# Set number of runs
runs = 1

# Set mini-batch sizes
batchsizes = [7, 8, 9] + range(10, 200, 10) + range(200, 501, 50)
# Log algos
# batchsizes = [10, 20, 50, 100, 150, 200, 400, 500]

# Set algorithms
backfilterconvalgos = ["cudnn"]
algods = ["cudnn"]  # Data gradient algorithm
algofwds = ["cudnn"]


benchmark = "VGG"
template = "VGG.dnntemplate"

datasetsize = 50000
date = datetime.datetime.today().strftime('%Y%m%d')
nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []

command = "./run_dnnmark_template.sh -b test_{} --template {}".format(benchmark, template)

logdir = "logs/{}/dnnmark_{}_microseries_{}/".format(host, benchmark, date)
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir", logdir

logfile_base = "dnnmark_{}_{}".format(host, benchmark)
for batch in batchsizes:
    for algod in algods:
        for algo in backfilterconvalgos:
            for algofwd in algofwds:
                algod_opt = " --algod {}".format(algod)
                logname = "{}_bs{}_algos{}-{}-{}".format(logfile_base, batch, algofwd, algo, algod)
                for run in range(runs):
                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print "file", logfile, "exists."
                    else:
                        command_pars = command + " -n {} --algo {} --algod {} --algofwd {} -d {}{}".format(
                            batch, algo, algod, algofwd, datasetsize, debuginfo_option)
                        task = {"comm": command_pars, "logfile": logfile, "batch": batch, "nvsmi": with_memory}
                        tasks.append(task)
                if nvprof:
                    iterations = 10
                    # print "BS: {}, Iterations: {}".format(batch,iterations)
                    nvlogname = "{}_iter{}".format(logname, iterations)
                    command_pars = command + " -n {} -d {} --algo {} --algod {} --algofwd {} --iter {} --warmup 0".format(
                        batch, datasetsize, algo, algod, algofwd, iterations)
                    logfile = os.path.join(logdir, "{}_%p.nvprof".format(nvlogname))
                    if os.path.isfile(logfile):
                        print "file", logfile, "exists."
                    else:
                        profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --csv --log-file {} {}".format(
                            logfile, command_pars)
                        task = {"comm": profcommand, "logfile": logfile, "batch": batch, "nvsmi": False}
                        tasks.append(task)

print "Have", len(tasks), "tasks"
gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=1, d=1, nvsmi=tasks[i]["nvsmi"], mode="dmon", debug=False)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"], "w+")
    f.write(tasks[i]["comm"] + "\n")
    f.write("b{}\n".format(tasks[i]["batch"]))
    f.write("GPU{}: {}\n".format(gpu, gpu_info))
    f.close()
    print time.strftime("[%d %H:%M:%S]"),
    multigpuexec.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
    print tasks[i]["logfile"]
    print "{}/{} tasks".format(i + 1, len(tasks))
    time.sleep(1)
