#!/bin/bash

source `dirname $0`/learn_large.sh

if [ $# != 1 ]; then
    device="cuda:0"
else
    device=$1
fi

datadir_root="/data/demonstrations/learn_hetero/21-30"
num_divide=1

learn ${datadir_root} ${device}
