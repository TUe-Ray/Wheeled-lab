#observarions

import torch
import cv2
import numpy as np
import math

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

def lidar_distances(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    """回傳 RayCaster 與 hit 點之間的距離 (m)。輸出 shape = (N_env, N_rays)."""
    sensor = env.scene[sensor_cfg.name]           # RayCaster 物件
    hits = sensor.data.ray_hits_w                 # (N, B, 3)
    origin = sensor.data.pos_w.unsqueeze(1)       # (N, 1, 3)
    dists = torch.norm(hits - origin, dim=-1)     # (N, B)
    return dists



def min_lidar_beam_angle(env,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    """
    Returns shape=(N_env,1) the angle (rad) of the ray
    whose distance is smallest – so the policy knows *where*
    the nearest obstacle is.
    """
    # 1) raw distances: (N_env, N_rays)
    dists = lidar_distances(env, sensor_cfg)
    # 2) which beam index is closest: (N_env,)
    idx   = torch.argmin(dists, dim=-1)

    # 3) query the sensor’s own pattern_cfg for angles
    sensor = env.scene.sensors[sensor_cfg.name]
    # pull the pattern_cfg from the sensor’s config (not the live object)
    hmin, hmax = sensor.cfg.pattern_cfg.horizontal_fov_range
    res        = sensor.cfg.pattern_cfg.horizontal_res

    # inclusive range from hmin to hmax in degrees
    deg_angles = torch.arange(hmin, hmax + 1e-6, res, device=env.device)
    angles     = deg_angles * (math.pi / 180.0)

    # 4) pick out the minimal-beam angle per env and return (N_env,1)
    return angles[idx].unsqueeze(-1)


### Commonly used observation terms with emprically determined noise levels

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
        #lidar_term = ObsTerm(
        #    func = lidar_distances,
        #    noise =Gnoise(mean =0.,std=0.02),
        #    clip=(0.0,50.0)

        #)

        # last_action_term = ObsTerm( # [m/s, (-1, 1)]
        #     func=mdp.last_action,
        #     clip=(-1., 1.), # TODO: get from ClipAction wrapper or action space
        # )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class NavObsCfg:
    """Give the policy x,y position + orientation + velocities."""

    @configclass
    class PolicyCfg(ObsGroup):
        root_pos_w_term = ObsTerm(
            func=mdp.root_pos_w,
            noise=None,
        )

        root_euler_xyz_term = ObsTerm(
            func=root_euler_xyz,
            noise=None,
        )

        base_lin_vel_term = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Gnoise(mean=0., std=0.5),
        )
        base_ang_vel_term = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Gnoise(std=0.4),
        )
        #lidar_term = ObsTerm(
        #    func=lidar_distances,
        #    noise=Gnoise(mean=0.0, std=0.02),
        #    clip=(0.0, 50.0)
        #)
        min_beam_angle_term = ObsTerm(
            func=min_lidar_beam_angle,
            noise=None,
            clip=(-math.pi, math.pi),
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
