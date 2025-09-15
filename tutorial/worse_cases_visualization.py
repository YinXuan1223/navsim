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

# Add the GTRS root to path
sys.path.append('/mnt/hdd5/Qiaoceng/navsim_workspace/GTRS')

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
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

def add_trajectory_to_bev_ax_custom(ax: plt.Axes, trajectory: np.ndarray, config: Dict[str, Any]) -> plt.Axes:
    """Add trajectory to BEV plot with custom styling"""
    poses = np.concatenate([np.array([[0, 0]]), trajectory[:, :2]])
    
    ax.plot(
        poses[:, 1],  # y coordinates
        poses[:, 0],  # x coordinates 
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"], 
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )
    return ax

def plot_bev_visualization(scene_loader, token: str, evaluation_results: pd.DataFrame, 
                          sorted_trajectories: np.ndarray, sorted_weights: np.ndarray, 
                          predict_trajectory, save_path: str, title_ground_truth: str, stage_one: bool = True):
    """Create and save BEV visualization"""
    
    scene = scene_loader.get_scene_from_token(token)
    scene_for_ground_truth = scene if stage_one else scene_loader.get_scene_from_token(scene.scene_metadata.corresponding_original_initial_token) 
    
    frame_idx = scene_for_ground_truth.scene_metadata.num_history_frames - 1
    ground_truth_trajectory = scene_for_ground_truth.get_future_trajectory()
    
    # Create subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    
    # Extract evaluation metrics
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

    # 1. Ground Truth BEV
    add_configured_bev_on_ax(ax1, scene_for_ground_truth.map_api, scene_for_ground_truth.frames[frame_idx])
    add_trajectory_to_bev_ax(ax1, ground_truth_trajectory, TRAJECTORY_CONFIG["human"])
    configure_bev_ax(ax1)
    configure_ax(ax1)
    ax1.set_title(title_ground_truth)

    # 2. Top 300 Predicted Trajectories
    frame_idx = scene.scene_metadata.num_history_frames - 1
    add_configured_bev_on_ax(ax2, scene.map_api, scene.frames[frame_idx])
    
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
    add_configured_bev_on_ax(ax3, scene.map_api, scene.frames[frame_idx])
    config_top100 = TRAJECTORY_PLOT_CONFIG.copy()
    config_top100['line_color'] = VISUALIZATION_COLORS[0]
    
    for i, trajectory in enumerate(sorted_trajectories[:100]):
        config_top100['line_color_alpha'] = sorted_weights[i] if i < len(sorted_weights) else 0.5
        add_trajectory_to_bev_ax_custom(ax3, trajectory, config_top100)
    
    configure_bev_ax(ax3)
    configure_ax(ax3)
    ax3.set_title("Top 100 Trajectories")

    # 4. Best Prediction Only
    add_configured_bev_on_ax(ax4, scene.map_api, scene.frames[frame_idx])
    if len(sorted_trajectories) > 0:
        add_trajectory_to_bev_ax(ax4, predict_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax4)
    configure_ax(ax4)
    ax4.set_title("Best GTRS Dense Prediction")

    plt.tight_layout()
    
    # Save the plot
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{token}.png", bbox_inches='tight', dpi=300)
        print(f"Saved visualization for token {token} to {save_path}/{token}.png")
    
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
