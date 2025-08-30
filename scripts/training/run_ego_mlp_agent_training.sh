TRAIN_TEST_SPLIT=mini

# Set environment variables for CPU training
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=""

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
experiment_name=training_ego_mlp_agent \
trainer.params.max_epochs=50 \
++trainer.params.accelerator=cpu \
+trainer.params.devices=1 \
train_test_split=$TRAIN_TEST_SPLIT \
