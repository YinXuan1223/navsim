TRAIN_TEST_SPLIT=mini
CHECKPOINT=/mnt/hdd5/Qiaoceng/navsim_workspace/model/LatentTransfuser_mini/0830.ckpt
CACHE_PATH=/mnt/hdd5/Qiaoceng/navsim_workspace/exp/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

export HYDRA_FULL_ERROR=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
worker=single_machine_thread_pool \
worker.max_workers=4 \
