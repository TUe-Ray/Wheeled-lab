import torch
import cv2
import numpy as np

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    AdditiveGaussianNoiseCfg as Gnoise,
)

from wheeledlab.envs.mdp import root_euler_xyz

### Commonly used observation terms with emprically determined noise levels

def lidar_distances(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    """回傳 RayCaster 與 hit 點之間的距離 (m)。輸出 shape = (N_env, N_rays)."""
    sensor = env.scene[sensor_cfg.name]           # RayCaster 物件
    hits = sensor.data.ray_hits_w                 # (N, B, 3)
    origin = sensor.data.pos_w.unsqueeze(1)       # (N, 1, 3)
    dists = torch.norm(hits - origin, dim=-1)     # (N, B)
    return dists

@configclass
class BlindObsCfg:
    """Default observation configuration (no sensors; no corruption)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # root_pos_w_term = ObsTerm( # meters
        #     func=mdp.root_pos_w,
        #     noise=Gnoise(mean=0., std=0.1),
        # )

        # root_euler_xyz_term = ObsTerm( # radians
        #     func=root_euler_xyz,
        #     noise=Gnoise(mean=0., std=0.1),
        # )

        base_lin_vel_term = ObsTerm( # m/s
            func=mdp.base_lin_vel,
            noise=Gnoise(mean=0., std=0.5),
        )

        base_ang_vel_term = ObsTerm( # rad/s
            func=mdp.base_ang_vel,
            noise=Gnoise(std=0.4),
        )
        lidar_term = ObsTerm(
            func = lidar_distances,
            noise =Gnoise(mean =0.,std=0.02),
            clip=(0.0,50.0)

        )

        # last_action_term = ObsTerm( # [m/s, (-1, 1)]
        #     func=mdp.last_action,
        #     clip=(-1., 1.), # TODO: get from ClipAction wrapper or action space
        # )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()