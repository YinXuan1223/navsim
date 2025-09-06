#!/bin/bash

# 訓練 Latent TransFuser 模型的腳本
TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_agent \
experiment_name=training_latent_transfuser_agent \
train_test_split=$TRAIN_TEST_SPLIT \
agent.config.latent=true \
agent.lr=1e-4
