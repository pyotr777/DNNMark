#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os
import datetime

gpus = range(0,1)
host = "mouse"
runs = 1
benchmark = "VGG"
template = "VGG.dnntemplate"
# batchsizes = range(10,110,10) + range(120,510,20)
batchsizes = [7,8,9] + range(10,50,2) + range(50,160,10) +  range(160,200,20) + range(200,500,50)
datasetsize = 50000
date = datetime.datetime.today().strftime('%Y%m%d')
backfilterconvalgos=[0,1,3,"cudnn"]
algod="1" # Data gradient algorithm

nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []
logdir = "logs/dnnmark_{}_microseries_{}/".format(benchmark,date)

command = "./run_dnnmark_template.sh -b test_{} --template {}".format(benchmark,template)
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir

logfile_base="dnnmark_{}_{}".format(host,benchmark)
for batch in batchsizes:
    #iterations = int(math.ceil(datasetsize/batch))
    for algo in backfilterconvalgos:
        # print "BS: {}, Iterations: {}".format(batch,iterations)
        logname = "{}_bs{}_algo{}_algod{}".format(logfile_base,batch,algo,algod)
        for run in range(runs):
            logfile = os.path.join(logdir,"{}_{:02d}.log".format(logname,run))
            if os.path.isfile(logfile):
                print "file",logfile,"exists."
            else:
                command_pars = command+" -n {} --algo {} --algod {} -d {}{}".format(
                    batch,algo,algod,datasetsize,debuginfo_option)
                task = {"comm":command_pars,"logfile":logfile,"batch":batch,"nvsmi":with_memory}
                tasks.append(task)
        if nvprof:
            iterations = 1
            # print "BS: {}, Iterations: {}".format(batch,iterations)
            nvlogname = "{}_iter{}".format(logname,iterations)
            command_pars = command+" -n {} --algo {} --algod {} -d {}".format(
            	batch,algo,algod,datasetsize)
            logfile = os.path.join(logdir,"{}_%p.nvprof".format(nvlogname))
            if os.path.isfile(logfile):
                print "file",logfile,"exists."
            else:
                profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --csv --log-file {} {}".format(logfile,command_pars)
                task = {"comm":profcommand,"logfile":logfile,"batch":batch,"nvsmi":False}
                tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=1,d=1,nvsmi=tasks[i]["nvsmi"],mode="dmon",debug=False)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write(tasks[i]["comm"]+"\n")
    f.write("b{}\n".format(tasks[i]["batch"]))
    f.write("GPU{}: {}\n".format(gpu,gpu_info))
    f.close()
    print time.strftime("[%d %H:%M:%S]")
    multigpuexec.runTask(tasks[i],gpu,nvsmi=tasks[i]["nvsmi"],delay=0,debug=False)
    print tasks[i]["logfile"]
    print "{}/{} tasks".format(i+1,len(tasks))
    time.sleep(0)


