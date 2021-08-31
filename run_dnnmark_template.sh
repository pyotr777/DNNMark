#!/bin/bash

# Wrapper API for DNNMark
# 2018-2021 (C) Peter Bryzgalov @ CHITECH Stair Lab

version="1.00"
IFS='' read -r -d '' usage <<USAGEBLOCK
Run DNNMark with parameters from CLI. v${version}.
Usage:
$(basename $0)  [-b <benchmark executable, default=test_composed_model>]
                [ --template - benchmark configuration template file]
                [ -d <dataset size> - number of samples in dataset, derives number of iterations from batch size and datasetsize]
                [ --warmup <int> - number of warmup iterations]
                [ --debug - debug info ]
                [ --help  - usage info ]
                [ --root <path> - DNNMark installation root]
                [ --iter <int> - number of FWD+BWD passes to measure time]
                [ --cachediterations - use faster iterations without data initialisation for FC, convolutional and BN layers]             
                [-n <number of images, batch size>]

                # Convolutional layer parameter
                [-c <number of channels in input images>]
                [-h <height of input images>]
                [-w <widht of input images>]
                [-k <number of filters, ouput channels>]
                [-s <size of filter kernel>]
                [-u <stride>]
                [-p <padding>]
                [ --algo <cudnnConvolutionBwdFilterAlgo_t> - cuDNN algorithm for backward filter convolution.
                    Can be set to "fft", "winograd", number from 0 to 6, "cudnn", "cudnnv7".]
                [ --bwd_filter_pref <fastest/no_workspace/specify_workspace_limit> - cuDNN backward filter algorithm selection preference]
                [ --bwd_data_pref <fastest/no_workspace/specify_workspace_limit> - cuDNN backward data algorithm selection preference]
                [ --fwd_pref <fastest/no_workspace/specify_workspace_limit> - cuDNN forward algorithm selection preference]
                [ --algod <cudnnConvolutionBwdDataAlgo_t> - cuDNN algorithm for backward data convolution.
                    Can be set to one of the following: "fft","winograd","winograd_nonfused","fft_tiling",0, 1, "cudnn", "cudnnv7". ]
                [ --algofwd <cudnnConvolutionFwdAlgo_t> - cuDNN algorithm for forward convolution.
                    Can be set to "fft", "winograd", number from 0 to 7, "cudnn", "cudnnv7".]
                [ --workspace <workspace size in bites for convolution functions>]
                [ --conv_mode <convolution/cross_correlation> - Convolution mode]
                [ --nopropagation - For the first layer do not calculate BWD data]
                
                # BN Layer parameters
                [ --bnmode <spacial/per_activation> - BN layer mode]
                [ --epsilon <float> - BN layer epsilon parameter]
                [ --exponentialAverageFactor <float> - BN layer factor used in the moving average computation as follows: runningMean = runningMean*(1-factor) + newMean*factor]

Configuration is saved to temporary file conf_tmp.dnnmark.
USAGEBLOCK

template="config_example/conf_convolution_block.dnntemplate"
config_file="conf_tmp.dnnmark"

# Defaults
default_conv_bwd_filter_pref="fastest"
default_conv_fwd_pref="fastest"
default_conv_bwd_data_pref="fastest"
default_workspace=250000000  # 250MB
N=64
C=3
H=32
W=32
K=128
S=3
U=1
P=1
BENCH="test_composed_model"
ITER=1
WARMUP="0"
debug=0
cachediterations=""
datasetsize=0
workspace=""
propagation=""
root="."
conv_mode="convolution"
MODE="spacial"
epsilon=0.001
exponentialAverageFactor=0.5

while test $# -gt 0; do
    case "$1" in
        --help)
            echo "$usage"
            exit 0
            ;;
        -n)
            N="$2";shift;
            ;;
        -c)
            C="$2";shift;
            ;;
        -h)
            H="$2";shift;
            ;;
        -w)
            W="$2";shift;
            ;;
        -k)
            K="$2";shift;
            ;;
        -s)
            S="$2";shift;
            ;;
        -u)
            U="$2";shift;
            ;;
        -p)
            P="$2";shift;
            ;;
        -b)
            BENCH="$2";shift;
            ;;
        -d)
            datasetsize="$2";shift;
            ;;
        --algo)
            CBFA="$2";shift;
            ;;
        --bwd_filter_pref)
            conv_bwd_filter_pref="$2";shift;
            ;;
        --bwd_data_pref)
            conv_bwd_data_pref="$2";shift;
            ;;
        --fwd_pref)
            conv_fwd_pref="$2";shift;
            ;;
        --algod)
            CBDA="$2";shift;
            ;;
        --algofwd)
            CFWA="$2";shift;
            ;;
        --workspace)
            workspace="workspace_size=$2"$'\n';shift;
            ;;
        --nopropagation)
            propagation="propagation=false"$'\n';
            ;;
        --conv_mode)
            conv_mode="$2";shift;
            ;;
        --iter)
            ITER="$2";shift;
            ;;
        --cachediterations)
            cachediterations=" --cachediterations";shift;
            ;;
        --warmup)
            WARMUP="$2";shift;
            ;;
        --debug)
            debug="$2";shift;
            ;;
        --template)
            template="config_example/$2.dnntemplate";shift;
            ;;
        --root)
            root="$2";shift;
            ;;
        --bnmode)
            MODE="$2";shift;
            ;;
        --epsilon)
            epsilon="$2";shift;
            ;;
        --exponentialAverageFactor)
            exponentialAverageFactor="$2";shift;
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

echo "Running DNNMark with CLI parameters. v${version}."

if [ $CBFA ];then
    CUDNN_CBFA="algo=$CBFA"$'\n'  # Insert new line; inside double quotes not expanded.
    echo "$CBFA"
    if [ "$CBFA" = "cudnn" ];then
        echo "algo is cudnn"
        if [ ! $conv_bwd_filter_pref ];then
            conv_bwd_filter_pref=$default_conv_bwd_filter_pref
            echo "FWD pref set to $bwd_filter_pref"
        fi
    fi
fi

if [ $CBDA ];then
    CUDNN_CBDA="algod=$CBDA"$'\n'
fi

if [ $CFWA ];then
    CUDNN_CFWA="algofwd=$CFWA"$'\n'
fi

divide_ceil() {
    echo "($1 + $2 - 1)/$2" | bc
}

# Calculate number of iterations from BS ($N) and dataset size
if [ $datasetsize -gt 0 ]; then
    echo "datasetsize=$datasetsize"
    ITER=$(divide_ceil $datasetsize $N)
fi

echo "Using template $template"
# Remove last '/' in root path.
# root=$(echo $root | sed 's/\(.*\)\//\1/')
conf="$(echo EOF;cat ${root}/${template};echo EOF)"

eval "cat <<$conf" >${config_file}
echo "--- Configuration file (${config_file}) ---"
cat $config_file
echo "-------------------------------------------------"
echo "Benchmark: $BENCH"
echo "Iterations:$ITER"
com="${root}/build/benchmarks/${BENCH}/dnnmark_${BENCH} -config ${config_file} --warmup $WARMUP --iterations ${ITER}${cachediterations} --debuginfo $debug"
echo $com
$com
