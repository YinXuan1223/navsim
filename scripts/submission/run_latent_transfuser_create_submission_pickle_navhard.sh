TEAM_NAME="hehehe"
AUTHORS="QiaocengYinxuan"
EMAIL="leevicky931223@gmail.com"
INSTITUTION="nycu"
COUNTRY="Taiwan"

TRAIN_TEST_SPLIT=navhard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
experiment_name=submission_transfuser_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
agent.checkpoint_path=$NAVSIM_DEVKIT_ROOT/model/0830.ckpt \
agent.config.latent=true
