#!/usr/bin/env python

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

import multigpuexec
import time
import os

gpus = range(0,1)
runs = 10
# batchsizes = range(10,110,10) + range(120,510,20)
# batchsizes = [7,8,9] + range(10,50,2) + range(50,160,10) +  range(160,200,20) + range(200,500,50)
batchsizes = [7,8,9,10,12,15,20,25,50,55,60,90,100,160,170,180,190,200,300,400,500]

# List of convolutional layer configurations
# conv_sizes = [256, 512]
# channels_sizes = [256, 512] # Number of channel in input data
backfilterconvalgos=[0,1,2,3,5,"cudnn"]
# VGG model convolution configs
configs = [(2,512,512),(4,512,512),(4,256,512),(8,256,256),(8,128,256),(16,128,128),(16,64,128),(32,64,64),(32,3,64)]
nvprof = False
with_memory = False
debuginfo = False
debuginfo_option = ""
if debuginfo:
    debuginfo_option = " --debug"
tasks = []
logdir = "logs/dnnmark_convmodel_microseries/"
iterations = 1
command = "nvprof --profile-child-processes --unified-memory-profiling off --csv ./run_dnnmark_template.sh"
if not os.path.exists(logdir):
    os.makedirs(logdir)
print "Logdir",logdir

logfile_base="dnnmark_mouse_convmodel"
for config in configs:
    imsize,channels,conv = config
# for imsize in imsizes:
#     for channels in channels_sizes:
#         for conv in conv_sizes:
    for batch in batchsizes:
        for algo in backfilterconvalgos:
            logname = "{}_imsize{}_channels{}_conv{}_bs{}_algo{}".format(logfile_base,imsize,channels,conv,batch,algo)
            for run in range(runs):
                logfile = os.path.join(logdir,"{}_{:02d}.log".format(logname,run))
                if os.path.isfile(logfile):
                    print "file",logfile,"exists."
                else:
                    command_pars = command+" -c {} -n {} -k {} -w {} -h {} --algo {} --iter {}{}".format(
                        channels,batch,conv,imsize,imsize,algo,iterations,debuginfo_option)
                    task = {"comm":command_pars,"logfile":logfile,"batch":batch,"conv":conv,"nvsmi":with_memory}
                    tasks.append(task)
            if nvprof:
                nvlogname = "{}_iter{}".format(logname,iterations)
                command_pars = command+" -c {} -n {} -k {} -w {} -h {} --algo {} --iter {}".format(channels,batch,conv,imsize,imsize,algo,iterations)
                logfile = os.path.join(logdir,"{}_%p.nvprof".format(nvlogname))
                if os.path.isfile(logfile):
                    print "file",logfile,"exists."
                else:
                    profcommand = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --csv --log-file {} {}".format(logfile,command_pars)
                    task = {"comm":profcommand,"logfile":logfile,"batch":batch,"conv":conv,"nvsmi":False}
                    tasks.append(task)

print "Have",len(tasks),"tasks"
gpu = -1
for i in range(0,len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu+1,c=1,d=1,nvsmi=tasks[i]["nvsmi"],mode="dmon",debug=False)
    gpu_info = multigpuexec.getGPUinfo(gpu)
    f = open(tasks[i]["logfile"],"w+")
    f.write(tasks[i]["comm"]+"\n")
    f.write("b{} conv{}\n".format(tasks[i]["batch"],tasks[i]["conv"]))
    f.write("GPU{}: {}\n".format(gpu,gpu_info))
    f.close()
    print time.strftime("%d,%H:%M:%S")
    multigpuexec.runTask(tasks[i],gpu,nvsmi=tasks[i]["nvsmi"],delay=0,debug=False)
    print tasks[i]["logfile"]
    print "{}/{} tasks".format(i+1,len(tasks))
    time.sleep(0)


