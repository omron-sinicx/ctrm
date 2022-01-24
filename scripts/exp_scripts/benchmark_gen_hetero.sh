#!/bin/bash

set -e

function gen() {
    benchmark_name=$1
    num_agents_min=$2
    num_agents_max=$3
    obs_num=$4
    max_speeds_cands=$5
    rads_cands=$6

    rootdir="/data/benchmark/"${benchmark_name}
    rm -rf ${rootdir}
    python `dirname $0`/../benchmark_gen.py --config-name=benchmark_hetero \
           instance.num_agents_min=${num_agents_min} \
           instance.num_agents_max=${num_agents_max} \
           instance.obs_num=${obs_num} \
           instance.max_speeds_cands=${max_speeds_cands} \
           instance.rads_cands=${rads_cands} \
           rootdir=${rootdir}
}

# name, min agents, max agents, obstacles, speeds, rads
gen homo-basis 21 30 10 "\"0.03125\"" "\"0.015625\""
gen homo-wo-obs 21 30 0 "\"0.03125\"" "\"0.015625\""
gen homo-many-obs 21 30 20 "\"0.03125\"" "\"0.015625\""
gen hetero 21 30 10 \
    "\"0.03125,0.0390625,0.046875\"" \
    "\"0.015625,0.01953125,0.0234375\""
gen homo-more-agents 31 40 10 "\"0.03125\"" "\"0.015625\""
gen homo-many-agents 41 50 10 "\"0.03125\"" "\"0.015625\""
