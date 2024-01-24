#!/usr/bin/env python3

# Prepares and runs multiple tasks on multiple GPUs: one task per GPU.
# Waits if no GPUs available. For GPU availability check uses "nvidia-smi dmon" command.

# For each MBS including last smaller MBS use fixed number of iterations

# 2018-2020 (C) Peter Bryzgalov @ CHITECH Stair Lab

import time
import os
import datetime
from datetime import timedelta
import math
import argparse
import pandas as pd
import sys

# sys.path.append('../lib')
from lib import lib3
import multigpuexec
import subprocess

version = '5.0b'


def main(args_main):
    print(
        'Running fixed number of iterations for each MBS and convolution configuration using test_fwd_conv and test_bwd_conv benchmarks.'\
        '\nScript v{}.'.format(version)
    )

    start_time = time.strftime("[%d %H:%M:%S]")

    # Set GPU range
    gpus = [0]

    # Change hostname
    host = multigpuexec.getHostname()  # "mouse.cont"

    template = "conv_alone_config"

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host, help="Host name. Default is {}.".format(host))
    parser.add_argument("--dir", "-d", default="./logs",
                        help="Path to the top level of dnnmark logs directories.")
    parser.add_argument(
        "--logdir", default=None, help=
        "Path to the log directory. If the case this options is set, no need to provide --dir option.")
    parser.add_argument("--runs", "-r", type=int, default=1,
                        help="Number of runs for each configuration and mini-batch size")
    parser.add_argument(
        "--profiling", choices=['off', 'profile', 'trace'], default='off', help=
        "Run nvprof (profile) or nsys (trace) for each configuration and mini-batch size. Valid choices: off (default), profile, trace."
    )
    parser.add_argument('--date', default=None, help='Set date for the logs path.')
    parser.add_argument(
        "--template", default=template,
        help="Configuration file template with .dnntemplate extension. Default: {}".format(template))
    parser.add_argument("--policy", default='chainer',
                        help='Convolution algorithms selection policy: chainer/pytorch.')
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Run DNNMark with --debuginfo option.")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Run warmup untill the target % of max Hz is reached.")
    parser.add_argument("--fastiterations", action="store_true",
                        help="Use faster iterations with data reuse")
    parser.add_argument("--detailedtime", action="store_true", default=False,
                        help="Print time for each operation on each iteration")
    parser.add_argument(
        "--convconfigs", default=None, type=str, nargs='*', help=
        "Convolutional layers configurations in hyphen-sepated WHCKSPUD format, e.g. 224-224-3-16-3-1-2-1"
    )
    parser.add_argument(
        "--cnns", type=str, default=None, nargs='*', help=
        "CNNs names.Can be set to 'all'. Benchmark all CNNs from the file convconfigs/<policy>_<cnns>_<dataset>.csv"
    )
    parser.add_argument(
        "--ignore-cnns", type=str, default=None, nargs='*',
        help="Opposite effect to the --cnns option: exclude these CNNs from benchmarking.")
    parser.add_argument("--dataset", default="imagenet",
                        help="Dataset name (imagenet, cifar, ...). Default: imagenet.")
    parser.add_argument("--gpus", type=int, nargs='*', default=None, help="GPU numbers")
    parser.add_argument("--iter", default=10, help="Use fixed number of iterations per epoch.")
    parser.add_argument("--datasetsize", default=50000, type=int,
                        help="Size of the training dataset. Defines number of iterations per one epoch.")
    parser.add_argument("--benchmarks", default=['test_fwd_conv_multibs', 'test_bwd_conv_multibs'],
                        nargs='*', type=str, help="DNNMark benchmark(s) (as a Python list).")
    parser.add_argument("--mbs", type=int, default=None, nargs='*',
                        help="Space-separated list of mini-batch sizes.")
    parser.add_argument(
        "--mbsrange", type=str, default=None, help=
        "Use all MBS from the range (inclusive on both ends). Provide two values separated with dash: <start>-<end>"
    )
    parser.add_argument("--dnnmark", default="/DNNMark", help="Driectory with a built DNNMark.")
    parser.add_argument('--maxusage', type=int, default=20,
                        help="GPU utilization threshold to consider GPU is free.")
    parser.add_argument('--maxtemp', type=int, default=65.,
                        help="Maximum GPU temperature. Use for running tasks on cooler GPUs.")
    parser.add_argument('--taskgroup', type=int, default=1,
                        help="Number of tasks executed consecutively.")
    parser.add_argument("--text", "-t", default="", help="Notes to save to README file.")
    parser.add_argument('--parse', action='store_true', default=False, help='Parse time logs')
    parser.add_argument("--cudnnlogs", action="store_true", default=False, help="Record cuDNN API logs.")
    args = parser.parse_args(args_main)

    if args.profiling == 'profile':
        profile_command = "nvprof -u s --profile-api-trace none --unified-memory-profiling off --profile-child-processes --profile-from-start off --csv  --log-file"
    elif args.profiling == 'trace':
        profile_command = 'nsys profile -t cuda,cudnn,nvtx,cublas -o'
    else:
        profile_command = None

    # Change hostname
    host = multigpuexec.getHostname()
    if args.host:
        host = args.host

    if args.iter is not None:
        print("Using {} iteration in epoch for each mini-batch size".format(args.iter))

    gpus = [0]
    if args.gpus is not None:
        gpus = args.gpus
    print("GPUs: {}".format(','.join([str(gpu) for gpu in gpus])))

    if args.template:
        template = args.template

    # Set number of runs
    runs = list(range(args.runs))
    print("{} runs".format(args.runs))

    batchsizes = None
    # Set mini-batch sizes
    if args.mbs is None:
        if args.mbsrange is not None:
            if '-' in args.mbsrange:
                mbs_range = args.mbsrange.split('-')
            elif ',' in args.mbsrange:
                mbs_range = args.mbsrange.split(',')
            else:
                print("Not correct format for mbsrange options. Provide two values separated with dash.")
                print(f"Received: {args.mbsrange}")
                raise ValueError("Incorrect mbsrange format.")
            try:
                mbsstart = int(mbs_range[0])
                mbsend = int(mbs_range[1])
            except ValueError as e:
                print(e)
                print("Incorrect format for mbsrange. Provide two values separated with dash.")
                raise e

            batchsizes = range(mbsstart, mbsend + 1)
    else:
        batchsizes = args.mbs
    print("Mini-batch sizes: {}".format(lib3.list2shortstring(batchsizes)))

    configs_path = "convconfigs"
    configs = []
    if args.cnns is not None:
        for cnn in args.cnns:
            csvfile = '{}/{}_{}_{}.csv'.format(configs_path, args.policy, cnn.lower(),
                                               args.dataset.lower())
            print("Reading {}".format(csvfile))
            if not os.path.exists(csvfile):
                print("Not found {}".format(csvfile))
                print("Check --cnns and --dataset options and check configuration files in {}/".format(
                    configs_path))
                sys.exit(1)
            config = pd.read_csv(csvfile)
            config.loc[:, 'nopropagation'] = False
            if args.model != 'all':
                config.loc[0, 'nopropagation'] = True
            configs.append(config)

        # Merge configs for individual CNNs into one Dataframe
        configs = pd.concat(configs, ignore_index=True)
    else:
        # Read convolution parameters from args.convconfigs
        # Convert to a dataframe
        configs = pd.DataFrame(
            columns=[
                'image height', 'image width', 'input channels', 'output channels', 'kernel size',
                'padding', 'stride', 'delation'
            ], data=[x.split('-') for x in args.convconfigs])
        configs.loc[:, 'nopropagation'] = False

    # Drop duplicate layer configurations
    configs.drop_duplicates(inplace=True)
    # Strip whitespaces from column names
    configs.rename(columns=lambda x: x.strip(), inplace=True)

    print("Layer Configurations")
    print(configs.head())

    # Set algorithm combinations
    algo_configs = {
        # "all-workspace10MB":
        "chainer":
        "--algofwd cudnn --algo cudnn --algod cudnn --fwd_pref specify_workspace_limit --bwd_filter_pref specify_workspace_limit --bwd_data_pref specify_workspace_limit --workspace 10000000",
        # "tf": "--algofwd auto --algo auto --algod auto"
        "pytorch": "--algofwd cudnnv7 --algo cudnnv7 --algod cudnnv7 --conv_mode cross_correlation"
    }
    algoconf = algo_configs[args.policy]

    datestr = ''
    if args.date:
        datestr = args.date
    else:
        # Use today's date or change to existing logs directory name
        datestr = datetime.datetime.today().strftime('%Y%m%d')
    print("Using date {}".format(datestr))

    with_memory = False
    debuginfo_option = ""
    if args.debug:
        debuginfo_option = " --debug 1"

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
    if args.warmup > 0:
        command += " --warmup {}".format(args.warmup)

    if args.fastiterations:
        command += " --cachediterations"
    if args.detailedtime:
        command += " --detailedtime"

    logfolder = "batchseries"
    if args.cudnnlogs:
        logfolder = "algoseries"
    logdir = None
    if args.logdir is None:
        logroot = "./logs/"
        if args.dir:
            logroot = args.dir
        logdir = os.path.join(logroot, host, logfolder,
                              "dnnmark_{}_ConvSeries_{}/".format(args.policy, datestr))
    else:
        logdir = args.logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print("Logdir", logdir)

    iterations = int(args.iter)
    if args.cudnnlogs:
        # iterations = 1
        os.environ["CUDNN_LOGINFO_DBG"] = "1"
        cudnnlogcounter = 0
        cudnnlogfilepattern = r"cudnnAPI_[0-9]+.*\.log"

    # Save --text argument to README
    if args.text:
        filename = os.path.join(logdir, "README")
        with open(filename, "w") as f:
            text = args.text
            if '__' in text:
                text = text.replace('__', ' ')
            f.write(text)
            print("Saved {}".format(filename))

    benchmarks = None
    if args.benchmarks is not None:
        # benchmarks = ast.literal_eval(args.benchmarks)
        benchmarks = args.benchmarks
    print("Using benchmark(s) {}".format(' '.join(benchmarks)))

    tasks = []
    if profile_command is None:
        for run in runs:
            other_options = algoconf
            batch = ','.join([str(b) for b in batchsizes])
            for benchmark in benchmarks:
                logfile_base = "dnnmark_{}_{}_{}".format(host, template.replace('_', ''),
                                                         benchmark.replace('_', ''))

                for _, config in configs.iterrows():
                    if 'cnn' in config:
                        config_cnns = config['cnn'].split(',')
                        if args.cnns is not None:
                            # Run only CNNs from args.cnns
                            # Current layer shape present in CNNs:
                            configfound = False
                            for cnn in config_cnns:
                                if cnn in args.cnns:
                                    configfound = True
                                    break
                            if not configfound:
                                # None of config_cnns are in --cnns option
                                print(f"Skipping configuration for {config['cnn']}")
                                continue
                        if args.ignore_cnns is not None:
                            # Ignore CNNs from args.ignore_cnns
                            ignore = True
                            for cnn in config_cnns:
                                if cnn not in args.ignore_cnns:
                                    ignore = False
                                    break
                            if ignore:
                                print(f"Skipping configuration for {config['cnn']}")
                                continue

                    H = config['image height']
                    W = config['image width']
                    C = config['input channels']
                    K = config['output channels']
                    S = config['kernel size']
                    P = config['padding']
                    U = config['stride']
                    D = config['delation']  # Not used TODO: add delation parameter to DNNMark
                    noprop = ''
                    if config['nopropagation'] == True:
                        noprop = " --nopropagation "
                    logname = "{}_shape{W}-{H}-{C}-{K}-{S}-{P}-{U}-{D}_algos{algos}".format(
                        logfile_base, H=W, W=H, C=C, K=K, S=S, P=P, U=U, D=D, algos=args.policy)

                    logfile = os.path.join(logdir, "{}_{:02d}.log".format(logname, run))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        # TODO: Add delation parameter after it is supported by DNNMark
                        command_pars = command + " -b {benchmark} -w {W} -h {H} -c {C} -k {K} -s {S} " \
                            "-p {P} -u {U} -n {batch} --iter {iter} {other}{noprop} {debug}".format(
                            benchmark=benchmark, W=W, H=H, C=C, K=K, S=S, P=P, U=U,
                            batch=batch, iter=iterations, other=other_options,
                            noprop=noprop, debug=debuginfo_option)

                        if args.cudnnlogs:
                            # cuDNN API logging
                            cudnnlogfile = "{}_b{}_{:04d}.log".format("cudnnAPI_%Y%m%d", batch,
                                                                      cudnnlogcounter)
                            cudnnlogcounter += 1
                            cudnnlogpath = os.path.join(logdir, cudnnlogfile)
                            os.environ["CUDNN_LOGDEST_DBG"] = cudnnlogpath
                            print("Writing cuDNN logs to {}".format(cudnnlogpath))
                            multigpuexec.message(command_pars)
                            with open(logfile, 'a') as f:
                                result = subprocess.run(command_pars.split(' '), stdout=f,
                                                        stderr=subprocess.PIPE)
                            time.sleep(1)
                        else:
                            # ordinary benchmarking
                            task = {"comm": command_pars, "logfile": logfile, "nvsmi": with_memory}
                            tasks.append(task)

    if profile_command is not None:
        for run in runs:
            other_options = algoconf
            batch = ','.join([str(b) for b in batchsizes])
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
                    D = config['delation']
                    noprop = ''
                    if config['nopropagation'] is not None:
                        noprop = " --nopropagation "
                    iterations = 10
                    logname = "{}_shape{W}-{H}-{C}-{K}-{S}-{P}-{U}-{D}_algos{algos}".format(
                        logfile_base, W=W, H=H, C=C, K=K, S=S, P=P, U=U, D=D, algos=args.policy)
                    nvlogname = "{}_iter{}_{:02d}".format(logname, iterations, run)
                    logfile = os.path.join(logdir, "{}_%p.nvprof".format(nvlogname))
                    if os.path.isfile(logfile):
                        print("file", logfile, "exists.")
                    else:
                        command_pars += " -w {W} -h {H} -c {C} -k {K} -s {S} -p {P} -u {U} -d {D} -n {batch} --iter {iter} {other}{noprop} --warmup 0".format(
                            W=W, H=H, C=C, K=K, S=S, P=P, U=U, D=D, batch=batch, iter=iterations,
                            other=other_options, noprop=noprop)
                        profcommand = "{} {} {}".format(profile_command, logfile, command_pars)
                        task = {"comm": profcommand, "logfile": logfile, "nvsmi": False}
                        tasks.append(task)

    print("Have", len(tasks), "tasks")
    gpu = -1
    start = time.perf_counter()
    start_datetime = datetime.datetime.now()
    lapstart = start
    i = 0
    while i < len(tasks):
        gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=3, d=1, nvsmi=tasks[i]["nvsmi"],
                                          mode="dmon", maxusage=args.maxusage, maxtemp=args.maxtemp,
                                          debug=False)
        os_info = multigpuexec.getOSinfo()

        for k in range(0, args.taskgroup):
            laptime = time.perf_counter() - lapstart
            totaltime = time.perf_counter() - start
            lapstart = time.perf_counter()
            done_ratio = (i + 1) / len(tasks)
            timepertask = totaltime / (i + 1)
            estimated_time_total = timepertask * len(tasks)
            estimated_finish = start_datetime + datetime.timedelta(seconds=estimated_time_total)
            print(
                "task {}/{} [{:.1f}%] laptime/from start {}/{}, estimated left {:.0f}s, estimated finish {}"
                .format(i + 1, len(tasks), done_ratio * 100., timedelta(seconds=laptime),
                        timedelta(seconds=totaltime), estimated_time_total - totaltime,
                        estimated_finish))
            f = open(tasks[i]["logfile"], "w+", buffering=-1)
            print(f"Logfile {tasks[i]['logfile']}")
            cpu_info = multigpuexec.getCPUinfo()
            gpu_info = multigpuexec.getGPUinfo(
                gpu, query=
                "name,memory.total,memory.free,pstate,temperature.gpu,clocks.mem,clocks.sm,clocks_throttle_reasons.active"
            )
            f.write("command:{}\n".format(tasks[i]["comm"]))
            f.write("GPU{}: {}\n".format(gpu, gpu_info))
            f.write("{}\n".format(cpu_info))
            f.write("{}\n".format(os_info))
            # Run parameters
            for par in [
                    'iter', 'fastiterations', 'warmup', 'detailedtime', 'maxusage', 'maxtemp',
                    'taskgroup'
            ]:
                val = getattr(args, par)
                f.write(f"{par}: {val}\n")
            f.write(f"runConvSeriesMultiMBS.py ver.{version}\n")
            f.close()
            print(time.strftime("[%d %H:%M:%S]"))
            if args.taskgroup == 1:
                multigpuexec.runTask(tasks[i], gpu, nvsmi=tasks[i]["nvsmi"], delay=0, debug=False)
            else:
                multigpuexec.runTaskCons(tasks[i], gpu, debug=False)
            i += 1
            if i >= len(tasks):
                break

    # Not using PIDs in runTaskCons, so multigpuexec.waitComplete() have no PIDs to check.
    multigpuexec.waitComplete()  # Wait for all running tasks to complete
    print("No more tasks to run.\nLogs are in {}".format(logdir))

    # Save file-flag that all tasks are finished.
    end_time = time.strftime("[%d %H:%M:%S]")
    filename = os.path.join(logdir, "DONE")
    with open(filename, "w") as f:
        f.write("Done {} tasks.\nTime: {} - {}\n".format(len(tasks), start_time, end_time))
        print("Saved {}".format(filename))

    if args.cudnnlogs:
        os.environ["CUDNN_LOGINFO_DBG"] = "0"
        # Parse
        command = "python3 ../pytorch/parseCudnnLogsConvolutions.py --log {patt} --dir {dir} --host {host}".format(
            patt=cudnnlogfilepattern, dir=logdir, host=host)
        multigpuexec.message("Parsing cuDNN logs\n{}".format(command))
        p = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
        try:
            while True:
                out = p.stdout.read()
                if out == '' and p.poll() != None:
                    break
                if out != '':
                    sys.stdout.write(out)
                    sys.stdout.flush()
        except Exception as e:
            print(e)
            print("Error reading line from stdout")
            print(type(line))
            print(line)
            sys.exit(1)
        p.stdout.close()
        p.wait()
        # Plot
        csv_file = os.path.join(logdir, "cudnnAPI_logs_aggregated.csv")
        command = "python3 ../pytorch/plotAlgoLogs.py --log {} --host {}".format(csv_file, host)
        multigpuexec.message("Plotting algorithms\n{}".format(command))
        result = subprocess.run(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        if args.parse:
            # Parsing time logs
            command = "python3 parseDNNMarkLogsMultiMBS.py --dir {} --model {} --av median".format(
                logdir, model)
            multigpuexec.message("Parsing log times\n{}".format(command))
            result = subprocess.run(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("All done.")


if __name__ == '__main__':
    main(sys.argv[1:])
