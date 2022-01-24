#!/bin/bash

set -e
time_limit=600

function run() {
    config_name=$1
    insdir=$2
    planner=$3
    list_num=$4
    for num in ${list_num[@]}
    do
        set -x
        python `dirname $0`/../eval.py \
               --config-name=${config_name} \
               insdir=${insdir} \
               roadmap.sample_rate_vs_diag_speed=${num} \
               planner=${planner} \
               planner.time_limit=${time_limit}
        set +x
    done
}

config="eval_large_rect"
planner="pp"
list_num="50 75 100"

####################################

insdir="/data/benchmark/homo-basis"
run ${config} ${insdir} ${planner} "${list_num}"

####################################

insdir="/data/benchmark/homo-wo-obs"
run ${config} ${insdir} ${planner} "${list_num}"

####################################

insdir="/data/benchmark/hetero"
run ${config} ${insdir} ${planner} "${list_num}"

####################################

insdir="/data/benchmark/homo-more-agents"
run ${config} ${insdir} ${planner} "${list_num}"

####################################

insdir="/data/benchmark/homo-many-obs"
run ${config} ${insdir} ${planner} "${list_num}"

####################################
