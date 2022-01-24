#!/bin/bash

set -e
time_limit=600

function run() {
    config_name=$1
    insdir=$2
    planner=$3
    set -x
    python `dirname $0`/../eval.py \
           --config-name=${config_name} \
           insdir=${insdir} \
           planner=${planner} \
           planner.time_limit=${time_limit}
    set +x
}

config="eval_large_spars"
planner="pp"

####################################

insdir="/data/benchmark/homo-basis"
run ${config} ${insdir} ${planner}

####################################

insdir="/data/benchmark/homo-wo-obs"
run ${config} ${insdir} ${planner}

####################################

insdir="/data/benchmark/hetero"
run "${config}_hetero" ${insdir} ${planner}

# Note: with multi-processing, but resulting in low success rate;
# hence the scores are excluded from the average plots

####################################

insdir="/data/benchmark/homo-more-agents"
run ${config} ${insdir} ${planner}

####################################

insdir="/data/benchmark/homo-many-obs"
run ${config} ${insdir} ${planner}

####################################
