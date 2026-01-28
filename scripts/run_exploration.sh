#!/bin/bash

sequential_num_run=20
prefix="ours_exploration"
exploration_uuid="q$(openssl rand -hex 2)"
seed=0
plan_model="Qwen/Qwen2.5-VL-7B-Instruct"

module_name="optimus1.main_exploration"

RANDOM_PORT=$((10000 + $RANDOM % 50000))
echo "Random port: $RANDOM_PORT"

for run in $(seq 1 $sequential_num_run)
do
    xvfb-run -a python app.py --port $RANDOM_PORT --seed $seed --plan_model "$plan_model" > /dev/null 2>&1 &
    sleep 3
    xvfb-run -a python -m $module_name server.port=$RANDOM_PORT env.times=1 prefix=$prefix exploration_uuid=$exploration_uuid exploration=True exp_num=$run seed=$((seed+run-1)) world_seed=$((seed+run-1)) plan_model="$plan_model"
    sleep 3
    python -m optimus1.util.clear_exploring_goals --exploration_uuid $exploration_uuid
    python -m optimus1.util.server_api --port $RANDOM_PORT
    sleep 3
done
"""
