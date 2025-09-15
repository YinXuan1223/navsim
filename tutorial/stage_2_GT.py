#!/usr/bin/env python3
"""
Script to save BEV visualizations for different token categories
Based on the analysis from plot_result.ipynb
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import lzma
# Add the GTRS root to path
sys.path.append('/mnt/hdd5/Qiaoceng/navsim_workspace/GTRS')

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from nuplan.common.actor_state.state_representation import TimePoint
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.config import TRAJECTORY_CONFIG

def setup_environment():
    """Set up environment variables and load configurations"""
    # Set environment variables explicitly
    os.environ["NUPLAN_MAPS_ROOT"] = "/mnt/hdd8/navsim_dataset/maps"
    os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
    os.environ["NUPLAN_DATA_STORE"] = "local"
    os.environ["OPENSCENE_DATA_ROOT"] = "/mnt/hdd8/navsim_dataset"
    os.environ["NAVSIM_EXP_ROOT"] = "/mnt/hdd5/Qiaoceng/navsim_workspace/exp"
    os.environ["NAVSIM_DEVKIT_ROOT"] = "/mnt/hdd5/Qiaoceng/navsim_workspace/GTRS"

    load_dotenv(override=True)
    
    print("Environment variables set successfully")
    
def load_data():
    """Load all required data and configurations"""
    
    # Also set the nuplan constants directly
    import nuplan.common.maps.nuplan_map.map_factory as map_factory
    map_factory.NUPLAN_MAPS_ROOT = "/mnt/hdd8/navsim_dataset/maps"
    
    # Print for debugging
    print(f"NUPLAN_MAPS_ROOT: {os.environ.get('NUPLAN_MAPS_ROOT')}")
    print(f"OPENSCENE_DATA_ROOT: {os.environ.get('OPENSCENE_DATA_ROOT')}")
    
    # Constants for data configuration
    DATA_SPLIT = "test"
    SCENE_FILTER_NAME = "navhard_two_stage"
    
    # Initialize Hydra configuration for scene filtering
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split/scene_filter", version_base=None)
    scene_filter_config = hydra.compose(config_name=SCENE_FILTER_NAME)
    scene_filter: SceneFilter = instantiate(scene_filter_config)
    
    # Setup data paths
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))
    navhard_scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{DATA_SPLIT}",
        openscene_data_root / f"sensor_blobs/{DATA_SPLIT}",
        scene_filter,
        openscene_data_root / "navhard_two_stage/sensor_blobs",
        openscene_data_root / "navhard_two_stage/synthetic_scene_pickles",
        sensor_config=SensorConfig.build_no_sensors(),
    )
    
    # Load GTRS Dense model predictions
    gtrs_dense_model_path = '/mnt/hdd5/Qiaoceng/navsim_workspace/model/GTRS/epoch19_navhard.pkl'
    with open(gtrs_dense_model_path, 'rb') as f:
        gtrs_dense_model_predictions = pickle.load(f)
    
    # Load trajectory vocabulary
    trajectory_vocab_path = '/mnt/hdd5/Qiaoceng/navsim_workspace/GTRS/traj_final/8192.npy'
    trajectory_vocabulary = np.load(trajectory_vocab_path)
    
    # Load evaluation results
    evaluation_results_path = "/mnt/hdd5/Qiaoceng/navsim_workspace/exp/train_gtrs_dense/2025.08.21.17.44.36/2025.08.21.18.36.35.csv"
    evaluation_results_df = pd.read_csv(evaluation_results_path)
    
    # Load submission file
    submission_file_path = "/mnt/hdd5/Qiaoceng/navsim_workspace/model/GTRS/submission_gtrs_dense.pkl"
    with open(submission_file_path, "rb") as f:
        submission_predictions = pickle.load(f)
    
    return (navhard_scene_loader, gtrs_dense_model_predictions, trajectory_vocabulary, 
            evaluation_results_df, submission_predictions)
    

def get_stage2_reference_trajectory(scene_loader, token: str):
    try:
        scene = scene_loader.get_scene_from_token(token)
        
        # Determine if it is Stage 2 - if there is corresponding_original_initial_token, it is Stage 2
        if not hasattr(scene.scene_metadata, 'corresponding_original_initial_token') or \
           scene.scene_metadata.corresponding_original_initial_token is None:
            print(f"Token {token} is not a Stage 2 scenario.")
            return None 
            
        log_name = scene.scene_metadata.log_name
        timestamp = scene.frames[scene.scene_metadata.num_history_frames - 1].timestamp

        ## Extract reference trajectory from metric cache
        stage2_cache_path = Path('/mnt/hdd5/Qiaoceng/navsim_workspace/exp/metric_cache_navhard_two_stage')
        
        # Search for corresponding cache file - match using log_name and token
        cache_file = None
        print('here!') 
        for scene_dir in stage2_cache_path.iterdir():
            if scene_dir.is_dir() and log_name in scene_dir.name:
                token_dir = None
                for sub_dir in scene_dir.rglob('*'):
                    if sub_dir.is_dir() and sub_dir.name == token:
                        token_dir = sub_dir
                        print(f"Matched token directory: {token_dir}")
                        break
                
                if token_dir:
                    print(f"Found token directory: {token_dir}")
                    pkl_file = token_dir / 'metric_cache.pkl'
                    if pkl_file.exists():
                        try:  # Validate if it is the correct cache
                            with lzma.open(pkl_file, 'rb') as f:
                                metric_cache = pickle.load(f)
                            if (hasattr(metric_cache, 'log_name') and  # Check if log_name and timestamp match
                                metric_cache.log_name == log_name and
                                hasattr(metric_cache, 'timepoint')):
                                # Simple time matching check
                                cache_time = metric_cache.timepoint.time_s
                                scene_time = timestamp / 1e6  # Convert to seconds
                                if abs(cache_time - scene_time) < 1.0:  # 1 second tolerance
                                    cache_file = pkl_file
                                    break
                        except Exception as e:
                            print(f"Error reading cache file {pkl_file}: {e}")
                            continue
                    
                if cache_file:
                    break
             
        if not cache_file:
            print(f"No corresponding cache file found for token: {token}")
            return None
        else:
            print(f"Found cache file: {cache_file} for token: {token}")
            
        # 讀取並處理軌跡
        with lzma.open(cache_file, 'rb') as f:
            metric_cache = pickle.load(f)
        
        traj = metric_cache.trajectory
        if not traj:
            return None
        
        # # 採樣軌跡點 - 從當前時間點開始採樣
        # start_time_us = traj.start_time.time_us
        # current_time_us = timestamp
        # print(f"Trajectory time range: {start_time_us} to {end_time_us}, current time: {current_time_us}")
        
        # # 找到與當前場景時間最接近的軌跡起始點
        # search_range_us = 2
        # search_start = max(start_time_us, current_time_us - search_range_us)
        # search_end = min(end_time_us, current_time_us + search_range_us)
        # print(f"Search start: {search_start}, Search end: {search_end}")
        
        # min_time_diff = float('inf')
        start_time_us = timestamp
        end_time_us = traj.end_time.time_us
        
        # for test_time_us in range(int(search_start), int(search_end), int(0.1 * 1e6)):
        #     time_diff = abs(test_time_us - current_time_us)
        #     if time_diff < min_time_diff:
        #         min_time_diff = time_diff
        #         best_start_time_us = test_time_us
        #         print(f"New best start time: {best_start_time_us} with time diff: {min_time_diff}")
        
        # if best_start_time_us is None:
        #     best_start_time_us = start_time_us
        
        # 每 0.5 秒採樣一次，總共 4 秒（8個點）
        sample_interval_us = 0.5
        
        time_points = []
        for i in range(8):  # 8個點
            time_us = start_time_us + i * sample_interval_us
            print(f"time_us: {time_us}, end_time_us: {end_time_us}")
            if time_us <= end_time_us:
                time_points.append(TimePoint(time_us))
            else:
                break
        print(f"Sampling time points (us): {[tp.time_us for tp in time_points]}")
        
        # 獲取每個時間點的狀態
        states = []
        for time_point in time_points:
            try:
                state = traj.get_state_at_time(time_point)
                states.append(state)
            except:
                break
        
        if len(states) < 2:
            return None
            
        # 獲取當前場景的 ego 起始位置
        current_ego_state = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status.ego_pose
        print(f"Current ego state: {current_ego_state}")
        
        # 獲取當前 ego 位置的 x, y, heading 坐標
        if hasattr(current_ego_state, 'x') and hasattr(current_ego_state, 'y'):
            ego_x, ego_y = current_ego_state.x, current_ego_state.y
            ego_heading = current_ego_state.heading if hasattr(current_ego_state, 'heading') else 0.0
        elif isinstance(current_ego_state, np.ndarray):
            ego_x, ego_y = current_ego_state[0], current_ego_state[1]
            ego_heading = current_ego_state[2] if len(current_ego_state) > 2 else 0.0
        else:
            ego_x, ego_y = current_ego_state[0], current_ego_state[1]
            ego_heading = current_ego_state[2] if len(current_ego_state) > 2 else 0.0
        
        # 轉換為相對於當前 ego 位置的軌跡（局部座標系）
        positions = []
        for i, state in enumerate(states):
            try:
                # 獲取軌跡點的絕對座標
                if hasattr(state, 'rear_axle') and hasattr(state.rear_axle, 'x'):
                    traj_x = state.rear_axle.x
                    traj_y = state.rear_axle.y
                elif hasattr(state, 'x') and hasattr(state, 'y'):
                    traj_x = state.x
                    traj_y = state.y
                elif isinstance(state, np.ndarray):
                    traj_x = state[0]
                    traj_y = state[1]
                else:
                    continue
                
                # 計算相對於 ego 的偏移（全局座標系）
                dx = traj_x - ego_x
                dy = traj_y - ego_y
                
                # 轉換到以 ego 為原點，ego 朝向為 x 軸的局部座標系
                cos_heading = np.cos(ego_heading)
                sin_heading = np.sin(ego_heading)
                
                # 旋轉變換：將全局座標轉換為局部座標
                local_x = cos_heading * dx + sin_heading * dy
                local_y = -sin_heading * dx + cos_heading * dy
                
                positions.append([local_x, local_y])
                
            except Exception as e:
                continue
        
        return np.array(positions)
        
    except Exception as e:
        return None


def main():
    print("Setting up environment...")
    setup_environment()
    
    print("Loading data...")
    (navhard_scene_loader, gtrs_dense_model_predictions, trajectory_vocabulary, 
        evaluation_results_df, submission_predictions) = load_data()
    
    # Extract tokens for different stages
    stage_one_tokens = list(submission_predictions['first_stage_predictions'][0].keys())
    stage_two_tokens = list(submission_predictions['second_stage_predictions'][0].keys())
    
    # Extract stage-specific results
    stage_one_metric_columns = [col for col in evaluation_results_df.columns if 'stage_one' in col]
    stage_one_results = evaluation_results_df[['token', 'valid', 'score'] + stage_one_metric_columns]
    stage_one_results = stage_one_results[stage_one_results['token'].isin(stage_one_tokens)]
    
    stage_two_metric_columns = [col for col in evaluation_results_df.columns if 'stage_two' in col]
    stage_two_results = evaluation_results_df[['token', 'valid', 'score'] + stage_two_metric_columns]
    stage_two_results = stage_two_results[stage_two_results['token'].isin(stage_two_tokens)]

    print(f'stage one result: {stage_one_results}')
    
    print('Getting reference trajectory for a stage 2 example...')
    positions = get_stage2_reference_trajectory(navhard_scene_loader, '4189825e20758be70')
    print(f'Reference trajectory shape: {positions.shape if positions is not None else None}')
    
if __name__ == "__main__":
    main()