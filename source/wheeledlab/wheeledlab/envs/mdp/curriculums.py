import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from collections.abc import Sequence
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)

def increase_reward_weight_over_time(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name : str,
        increase : float,
        episodes_per_increase : int = 1,
        max_increases: int = torch.inf,
        ) -> torch.Tensor:
    """
    Increase the weight of a reward term after some amount of given time in episodes.
    Default amount of time is one episode.
    Stops increasing the weight after `stop_after_n_changes` changes. Defaults to inf.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    num_increases = num_episodes // episodes_per_increase

    if num_increases > max_increases:
        return

    if env.common_step_counter % env.max_episode_length != 0:
        return

    # Only process at the beginning of an episode where increase should occur
    if (num_episodes + 1) % episodes_per_increase == 0:
        # Check if the reward term exists
        available = []
        # Attempt to retrieve term names via method or attribute
        if hasattr(env.reward_manager, 'term_names') and callable(env.reward_manager.term_names):
            available = env.reward_manager.term_names()
        else:
            try:
                available = [t.name for t in env.reward_manager.terms]
            except Exception:
                available = []

        if reward_term_name not in available:
            print(f"[increase_reward_weight_over_time] Warning: Reward term '{reward_term_name}' not found. Skipping.")
            return

        # Proceed to update the term weight
        try:
            term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        except ValueError as e:
            print(f"[increase_reward_weight_over_time] Error retrieving term cfg: {e}. Skipping.")
            return

        term_cfg.weight += increase
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)
