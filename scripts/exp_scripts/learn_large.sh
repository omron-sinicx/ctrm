#!/bin/bash

set -e
num_neighbors=15
fov_size=19
map_size=160
num_divide=3
use_back=0
batch_size=50
num_epochs=1000
without_comm=0

function learn() {
    datadir=$1
    device=$2
    python `dirname $0`/../train.py \
                datadir=${datadir} \
                format_input.use_k_neighbor=true \
                format_input.num_neighbors=${num_neighbors} \
                format_input.fov_encoder.map_size=${map_size} \
                format_input.fov_encoder.fov_size=${fov_size} \
                format_input.without_comm=${without_comm} \
                format_output.num_divide=${num_divide} \
                format_output.use_back=${use_back} \
                dataloader.batch_size=${batch_size} \
                num_epochs=${num_epochs} \
                intermediate.eval=false \
                device=${device}
}
