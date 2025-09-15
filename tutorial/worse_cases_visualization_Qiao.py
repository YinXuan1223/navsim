#!/usr/bin/env python3
'''
Rename + Stage 2 ground truth
'''

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
import lzma

# Add the GTRS root to path
sys.path.append('/mnt/hdd5/Qiaoceng/navsim_workspace/GTRS')

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_bev_ax, configure_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from nuplan.common.actor_state.state_representation import TimePoint

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

def calculate_scores(result):
    """Calculate trajectory scores"""
    time_to_collision_within_bound = np.exp(result['time_to_collision_within_bound'])
    ego_progress = np.exp(result['ego_progress'])
    lane_keeping = np.exp(result['lane_keeping'])
    scores = (
        0.01 * result['imi'] +
        0.1 * result['traffic_light_compliance'] +
        0.5 * result['no_at_fault_collisions'] +
        0.5 * result['drivable_area_compliance'] +
        0.5 * result['driving_direction_compliance'] +
        3.0 * np.log(5.0 * time_to_collision_within_bound +
                    5.0 * ego_progress +
                    2.0 * lane_keeping)
    )
    return scores

def normalize_score(scores):
    """Normalize trajectory scores"""
    scores = np.where(np.isfinite(scores), scores, np.nan)
    if np.all(np.isnan(scores)):
        return np.zeros_like(scores)
    min_score = np.nanmin(scores)
    max_score = np.nanmax(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    norm = (scores - min_score) / (max_score - min_score)
    norm = np.nan_to_num(norm, nan=0.0)
    return norm

def get_trajectory_predictions_for_token(token: str, trajectory_vocabulary, gtrs_dense_model_predictions) -> Tuple[np.ndarray, np.ndarray]:
    """Get sorted trajectory predictions for a token"""
    total_trajectory = np.concatenate((trajectory_vocabulary, gtrs_dense_model_predictions[token]['interpolated_proposal']), axis=0)
    
    scores = calculate_scores(gtrs_dense_model_predictions[token])
    weights = normalize_score(scores)
    sorted_indices = np.argsort(weights)[::-1]
    sorted_trajectories = total_trajectory[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    return sorted_trajectories, sorted_weights

# Visualization configurations
VISUALIZATION_COLORS: Dict[int, str] = {
    0: "#D10808",  # red
    1: "#000000",  # black
    2: "#2ECC71",  # green
    3: "#E38C47",  # orange
    4: "#8E44AD",  # purple
}

TRAJECTORY_PLOT_CONFIG = {
    "fill_color": VISUALIZATION_COLORS[0],
    "fill_color_alpha": 1.0,
    "line_color": VISUALIZATION_COLORS[0],
    "line_color_alpha": 1.0,
    "line_width": 0.3,
    "line_style": "-",
    "zorder": 3,
}

DRIVING_COMMAND_LABELS = ["LEFT", "STRAIGHT", "RIGHT", "UNKNOWN"]

def find_token_row_in_yaml(token: str, yaml_file_path: str = "/mnt/hdd5/Qiaoceng/navsim_workspace/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml") -> int:
    try:
        with open(yaml_file_path, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):  # 行數從1開始
            if token in line:
                return i
        return -1  # 沒找到
    except Exception as e:
        print(f"查找token行數時出錯: {e}")
        return -1

def get_stage2_reference_trajectory(scene_loader, token: str):
    try:
        scene = scene_loader.get_scene_from_token(token)
        
        # Determine if it is Stage 2 - if there is corresponding_original_initial_token, it is Stage 2
        if not hasattr(scene.scene_metadata, 'corresponding_original_initial_token') or \
           scene.scene_metadata.corresponding_original_initial_token is None:
            return None 
            
        log_name = scene.scene_metadata.log_name
        timestamp = scene.frames[scene.scene_metadata.num_history_frames - 1].timestamp

        ## Extract reference trajectory from metric cache
        # Construct metric cache path - use more precise search
        stage2_cache_path = Path('/mnt/hdd5/Qiaoceng/navsim_workspace/exp/metric_cache_stage_2')
        
        # Search for corresponding cache file - match using log_name
        cache_file = None
        for scene_dir in stage2_cache_path.iterdir():
            if scene_dir.is_dir() and log_name in scene_dir.name:
                for sub_dir in scene_dir.rglob('*'):
                    if sub_dir.is_dir():
                        pkl_file = sub_dir / 'metric_cache.pkl'
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
                            except:
                                continue
                if cache_file:
                    break
        
        if not cache_file:
            print(f"未找到匹配的 cache 檔案，token: {token}")
            return None
            
        # 讀取並處理軌跡
        with lzma.open(cache_file, 'rb') as f:
            metric_cache = pickle.load(f)
        
        traj = metric_cache.trajectory
        if not traj:
            return None
        
        # 採樣軌跡點 - 從當前時間點開始採樣
        start_time_us = traj.start_time.time_us
        end_time_us = traj.end_time.time_us
        current_time_us = timestamp * 1e6
        
        # 找到與當前場景時間最接近的軌跡起始點
        search_range_us = 2.0 * 1e6  # 搜索範圍：前後2秒
        search_start = max(start_time_us, current_time_us - search_range_us)
        search_end = min(end_time_us, current_time_us + search_range_us)
        
        min_time_diff = float('inf')
        best_start_time_us = None
        
        for test_time_us in range(int(search_start), int(search_end), int(0.1 * 1e6)):
            time_diff = abs(test_time_us - current_time_us)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_start_time_us = test_time_us
        
        if best_start_time_us is None:
            best_start_time_us = start_time_us
        
        # 每 0.5 秒採樣一次，總共 4 秒（8個點）
        sample_interval_us = int(0.5 * 1e6)
        
        time_points = []
        for i in range(8):  # 8個點
            time_us = best_start_time_us + i * sample_interval_us
            if time_us <= end_time_us:
                time_points.append(TimePoint(time_us))
            else:
                break
        
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

def add_trajectory_to_bev_ax_custom(ax: plt.Axes, trajectory: np.ndarray, config: Dict[str, Any], show_waypoints: bool = False) -> plt.Axes:
    """Add trajectory to BEV plot with custom styling"""
    poses = np.concatenate([np.array([[0, 0]]), trajectory[:, :2]])
    
    # 畫軌跡線
    ax.plot(
        poses[:, 1],  # y coordinates
        poses[:, 0],  # x coordinates 
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"], 
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )
    
    # 如果需要，顯示 waypoints
    if show_waypoints and len(poses) > 1:
        ax.scatter(
            poses[1:, 1],  # y coordinates (跳過起始點)
            poses[1:, 0],  # x coordinates (跳過起始點)
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            s=20,  # marker size
            zorder=config["zorder"] + 1,
            marker='o'
        )
    
    return ax

def plot_bev_visualization(scene_loader, token: str, evaluation_results: pd.DataFrame, 
                          sorted_trajectories: np.ndarray, sorted_weights: np.ndarray, 
                          predict_trajectory, save_path: str, title_ground_truth: str, stage_one: bool = True):
    """Create and save BEV visualization"""
    
    # Analyze scene structure for better understanding (optional - can be commented out)
    # analyze_scene_structure(scene_loader, token)
    
    scene = scene_loader.get_scene_from_token(token)
    
    if stage_one:
        # Stage 1: 使用相同場景
        scene_for_ground_truth = scene
        map_for_visualization = scene.map_api
        frame_idx_for_gt = scene.scene_metadata.num_history_frames - 1
        frame_idx_for_prediction = scene.scene_metadata.num_history_frames - 1
    else:
        # Stage 2: 使用原始場景的軌跡作為 ground truth，但用當前場景的地圖
        original_scene = scene_loader.get_scene_from_token(scene.scene_metadata.corresponding_original_initial_token)
        scene_for_ground_truth = original_scene
        map_for_visualization = scene.map_api  # 所有子圖都用 Stage 2 場景的地圖
        frame_idx_for_gt = original_scene.scene_metadata.num_history_frames - 1
        frame_idx_for_prediction = scene.scene_metadata.num_history_frames - 1
    
    ground_truth_trajectory = scene_for_ground_truth.get_future_trajectory()
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    
    # -1. Score and Metrics Info
    token_results = evaluation_results[evaluation_results['token'] == token].iloc[0]
    metrics_info = '\n'.join([f"{col}: {value:.4f}" if isinstance(value, float) else f"{col}: {value}" 
                             for col, value in token_results.items()])
    
    # Get driving command
    driving_command_idx = scene.get_agent_input().ego_statuses[-1].driving_command.argmax()
    driving_command = DRIVING_COMMAND_LABELS[driving_command_idx]
    metrics_info += f"\nDriving Command: {driving_command}"
    
    # Add metrics text
    plt.figtext(0.01, 1, metrics_info, wrap=True, horizontalalignment='left', 
                fontsize=10, color='black', verticalalignment='top')
    
    # 0. File name
    scene_token = scene.scene_metadata.scene_token
    current_frame_idx = scene.scene_metadata.num_history_frames - 1
    current_frame = scene.frames[current_frame_idx]
    timestamp = current_frame.timestamp
    
    if stage_one:
        meaningful_filename = f"{scene_token}_{timestamp}_{token}"
    else:   # Stage 2: 使用原始token + 行數 + 當前token的命名方式
        original_token = scene.scene_metadata.corresponding_original_initial_token
        row_num = find_token_row_in_yaml(token)
        if row_num != -1:
            meaningful_filename = f"{original_token}_{row_num:04d}_{token}"
        else:   # 如果找不到行數，回退到原來的命名方式
            meaningful_filename = f"{original_token}_norow_{token}"

    ego_progress_col = None
    for col in token_results.index:
        if 'ego_progress' in col:
            ego_progress_col = col
            break
    if ego_progress_col and token_results[ego_progress_col] < 0.5:
        meaningful_filename += "_LOW_EP"  # Add special marker for low ego progress

    # 1. Ground Truth BEV
    add_configured_bev_on_ax(ax1, map_for_visualization, scene.frames[frame_idx_for_prediction])
    
    if stage_one:  # 顯示真實的人類軌跡
        add_trajectory_to_bev_ax(ax1, ground_truth_trajectory, TRAJECTORY_CONFIG["human"])
    else:   # 只顯示參考軌跡，不顯示對應原始場景的軌跡
        stage2_reference_traj = get_stage2_reference_trajectory(scene_loader, token)
        if stage2_reference_traj is not None:
            add_trajectory_to_bev_ax_custom(ax1, stage2_reference_traj, TRAJECTORY_CONFIG["human"], show_waypoints=True)
        else:   # 如果無法提取參考軌跡，添加說明文字
            ax1.text(0.5, 0.5, 'No Reference Trajectory\nAvailable for Stage 2', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=12, color='red', weight='bold')
    
    configure_bev_ax(ax1)
    configure_ax(ax1)
    if not stage_one:
        ax1.set_title(f"{title_ground_truth}\n(Green: Stage2 Reference Trajectory - For Scoring)")
    else:
        ax1.set_title(title_ground_truth)

    # 2. Top 300 Predicted Trajectories
    add_configured_bev_on_ax(ax2, map_for_visualization, scene.frames[frame_idx_for_prediction])
    
    # Plot trajectories outside top 300 with black color first (background)
    config_black = TRAJECTORY_PLOT_CONFIG.copy()
    config_black['line_color'] = VISUALIZATION_COLORS[1]  # black
    config_black['line_color_alpha'] = 0.1  # very light
    for i, trajectory in enumerate(sorted_trajectories[300:]):
        add_trajectory_to_bev_ax_custom(ax2, trajectory, config_black)
    
    # Plot trajectories in tiers (top 300 only)
    if len(sorted_trajectories) > 200:
        config_tier3 = TRAJECTORY_PLOT_CONFIG.copy()
        config_tier3['line_color'] = VISUALIZATION_COLORS[4]  # purple
        for i, trajectory in enumerate(sorted_trajectories[200:300]):
            config_tier3['line_color_alpha'] = sorted_weights[200 + i] if 200 + i < len(sorted_weights) else 0.3
            add_trajectory_to_bev_ax_custom(ax2, trajectory, config_tier3)
    
    if len(sorted_trajectories) > 100:
        config_tier2 = TRAJECTORY_PLOT_CONFIG.copy()
        config_tier2['line_color'] = VISUALIZATION_COLORS[2]  # green
        for i, trajectory in enumerate(sorted_trajectories[100:200]):
            config_tier2['line_color_alpha'] = sorted_weights[100 + i] if 100 + i < len(sorted_weights) else 0.5
            add_trajectory_to_bev_ax_custom(ax2, trajectory, config_tier2)
    
    config_tier1 = TRAJECTORY_PLOT_CONFIG.copy()
    config_tier1['line_color'] = VISUALIZATION_COLORS[0]  # red
    for i, trajectory in enumerate(sorted_trajectories[:100]):
        config_tier1['line_color_alpha'] = sorted_weights[i] if i < len(sorted_weights) else 0.7
        add_trajectory_to_bev_ax_custom(ax2, trajectory, config_tier1)
    
    configure_bev_ax(ax2)
    configure_ax(ax2)
    total_trajs = len(sorted_trajectories)
    ax2.set_title(f'All {total_trajs} Trajectories (Red: 1-100, Green: 101-200, Purple: 201-300, Black: 301+)')

    # 3. Top 100 Trajectories Only
    add_configured_bev_on_ax(ax3, map_for_visualization, scene.frames[frame_idx_for_prediction])
    config_top100 = TRAJECTORY_PLOT_CONFIG.copy()
    config_top100['line_color'] = VISUALIZATION_COLORS[0]
    
    for i, trajectory in enumerate(sorted_trajectories[:100]):
        config_top100['line_color_alpha'] = sorted_weights[i] if i < len(sorted_weights) else 0.5
        add_trajectory_to_bev_ax_custom(ax3, trajectory, config_top100)
    
    configure_bev_ax(ax3)
    configure_ax(ax3)
    ax3.set_title("Top 100 Trajectories")

    # 4. Best Prediction Only
    add_configured_bev_on_ax(ax4, map_for_visualization, scene.frames[frame_idx_for_prediction])
    if len(sorted_trajectories) > 0:
        add_trajectory_to_bev_ax(ax4, predict_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax4)
    configure_ax(ax4)
    ax4.set_title("Best GTRS Dense Prediction")

    plt.tight_layout()
    
    # Save the plot with meaningful filename
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_filename = f"{meaningful_filename}.png"
        plt.savefig(f"{save_path}/{full_filename}", bbox_inches='tight', dpi=300)
        print(f"🎯 Saved visualization {save_path}/{full_filename}")
        
    plt.close()  # Close to save memory

def save_all_visualizations():
    """Main function to save all visualizations"""
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
    # Find worse case (EC, EP)
    stage_one_worse_conditions = (
        (stage_one_results['ego_progress_stage_one'] <= 1)
    )

    stage_two_worse_conditions = (
        (stage_two_results['ego_progress_stage_two'] <= 1)
    )

    # Find worse case score tokens
    stage_one_worse_case_tokens = list(stage_one_results[stage_one_worse_conditions]['token'])
    stage_two_worse_case_tokens = list(stage_two_results[stage_two_worse_conditions]['token'])

    print(f"Found {len(stage_one_worse_case_tokens)} worse_case_tokens for stage 1")
    print(f"Found {len(stage_two_worse_case_tokens)} worse_case_tokens for stage 2")

    
    # Save visualizations for each category
    '''
    # Stage 1 - Worse  case
    print("\nSaving Stage 1 worse case visualizations...")
    for i, token in enumerate(stage_one_worse_case_tokens):
        try:           
            sorted_trajectories, sorted_weights = get_trajectory_predictions_for_token(
                token, trajectory_vocabulary, gtrs_dense_model_predictions)
            
            plot_bev_visualization(
                scene_loader=navhard_scene_loader,
                token=token,
                evaluation_results=stage_one_results,
                sorted_trajectories=sorted_trajectories,
                sorted_weights=sorted_weights,
                predict_trajectory=gtrs_dense_model_predictions[token]['trajectory'],
                save_path='new_output/stage_one_worse_case',
                title_ground_truth="Human Trajectory (Stage 1)",
                stage_one=True
            )
        except Exception as e:
            print(f"Failed to save visualization for token {token}: {str(e)}")
    '''
    # Stage 2 - Worse  case
    print("\nSaving Stage 2 worse case visualizations...")
    for i, token in enumerate(stage_two_worse_case_tokens):
        try:   
            sorted_trajectories, sorted_weights = get_trajectory_predictions_for_token(
                token, trajectory_vocabulary, gtrs_dense_model_predictions)
            
            plot_bev_visualization(
                scene_loader=navhard_scene_loader,
                token=token,
                evaluation_results=stage_two_results,
                sorted_trajectories=sorted_trajectories,
                sorted_weights=sorted_weights,
                predict_trajectory=gtrs_dense_model_predictions[token]['trajectory'],
                save_path='new_output/stage_two_worse_case',
                title_ground_truth="Corresponding Scene (Stage 2)",
                stage_one=False
            )
        except Exception as e:
            print(f"Failed to save visualization for token {token}: {str(e)}")
    
    
    print("\n✅ All visualizations saved successfully!")

if __name__ == "__main__":
    save_all_visualizations()
