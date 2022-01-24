#!/bin/bash

set -e

config="eval_ctrm_ablation_learned_ind"
prob_uniform_sampling_after_goal=0.9
prob_uniform_bias=0

function run() {
    set -x
    pred_basename=$1
    python `dirname $0`/../eval.py \
            --config-name=${config} \
            roadmap.pred_basename=${pred_basename} \
            roadmap.prob_uniform_sampling_after_goal=${prob_uniform_sampling_after_goal} \
            roadmap.prob_uniform_bias=${prob_uniform_bias} \
            planner.time_limit=600
    set +x
}

run "/data/models/with_ind_k15/best"
run "/data/models/with_ind_k15_wo_comm/best"
run "/data/models/with_ind_k15_wo_ind/best"

# without random walk
prob_uniform_sampling_after_goal=0
prob_uniform_bias=1
run "/data/models/with_ind_k15/best"
