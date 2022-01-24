#!/bin/bash

set -e

function gen() {
    scenario_name=$1
    num_agents_min=$2
    num_agents_max=$3
    rootdir="/data/demonstrations/${scenario_name}/${num_agents_min}-${num_agents_max}"
    rm -rf ${rootdir}
    python `dirname $0`/../create_data.py --config-name=${scenario_name} \
           instance.num_agents_min=${num_agents_min} \
           instance.num_agents_max=${num_agents_max} \
           rootdir=${rootdir}
}

gen learn_hetero 21 30
