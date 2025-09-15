TRAIN_TEST_SPLIT=test
CACHE_PATH=/mnt/hdd5/Qiaoceng/navsim_workspace/exp/metric_cache_navhard_two_stage

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=human_agent \
experiment_name=human_agent \
traffic_agents=non_reactive \
metric_cache_path=$CACHE_PATH \
