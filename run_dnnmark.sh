#!/bin/bash

# API for DNNMark
# Provide benchmark name with -b parameter,
# and configuration file with --config parameter.
#
# 2018-2019 (C) Peter Bryzgalov @ CHITECH Stair Lab

usage=$(cat <<USAGEBLOCK
Run DNNMark with parameters from CLI.
Usage:
$(basename $0)  [-b <benchmark executable, default=test_composed_model>]
                [ --config <config_file.dnnmark> - DNNMark configuration file. Default=config_example/conf_convolution_block.dnnmark]
                [ --warmup - perform warmup runs before benchmarking]
                [ --debug - print debug info ]
                [ --help  - usage info ]

Configuration saved in temporary file conf_tmp.dnnmark
USAGEBLOCK
)

config_file="config_example/conf_convolution_block.dnnmark"
BENCH="test_composed_model"
debug=0
warmup=0

while test $# -gt 0; do
    case "$1" in
        --help)
            echo "$usage"
            exit 0
            ;;
        -b)
            BENCH="$2";shift;
            ;;
        --config)
            config_file="$2";shift;
            ;;
        --debug)
            debug=1
            ;;
        --warmup)
            warmup=1
            ;;
        --)
            shift
            break;;
        -*)
            echo "Unknown option $1";
            echo "$usage"
            exit 1
            ;;
        *)
            break;;
    esac
    shift
done

echo "./build/benchmarks/$BENCH/dnnmark_$BENCH -config $config_file -debuginfo $debug -warmup $warmup"
./build/benchmarks/"$BENCH"/dnnmark_"$BENCH" -config $config_file -debuginfo $debug -warmup $warmup



