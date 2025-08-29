TRAIN_TEST_SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_agent \
experiment_name=training_transfuser_agent \
train_test_split=$TRAIN_TEST_SPLIT \

# agent.batch_size=16 \
# agent.num_workers=8 \
