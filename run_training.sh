#! /bin/bash

function usage() {
    cat <<USAGE

    Usage: $0 [-c config] [-d dataset_path]

    Options:
	-c, --config:      Which config file to use
        -d, --dataset_path:  Path of the v-NICO-World-LL dataset
USAGE
    exit 1
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

SKIP_VERIFICATION=false
TAG=

while [ "$1" != "" ]; do
    case $1 in
    -c | --config)
        shift
        config=$1
        ;;
    -d | --dataset_path)
        shift 
	dataset_path=$1
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
    esac
    shift
done

echo "Start training using "$config ...
config_path=$(realpath $config)
dataset_path=$(realpath $dataset_path)
python src/train.py -c $config_path -d $dataset_path
