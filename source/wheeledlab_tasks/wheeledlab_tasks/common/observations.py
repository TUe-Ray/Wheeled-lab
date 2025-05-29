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


MAX_SPEED = 3.0

# def wheel_encoder(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
#     """
#     Returns a tensor of shape (N_envs, 2): [v_left, v_right] [m/s]
#     simply reading the joint velocities of the left & right wheels.
#     """
#     robot = env.scene[asset_cfg.name]               # Articulation
#     # find the same joint indices you used in your action term:
#     left_ids, _  = robot.find_joints(["front_left_wheel_joint", "rear_left_wheel_joint"])
#     right_ids, _ = robot.find_joints(["front_right_wheel_joint", "rear_right_wheel_joint"])
#     # joint_vel is (N_envs, n_joints)
#     joint_vel = robot.data.joint_vel               # rad/s
#     # convert from rad/s → m/s: v = ω * r
#     r = 0.05  # wheel radius in your SkidSteerActionCfg
#     v_left  = joint_vel[:, left_ids].mean(dim=-1) * r
#     v_right = joint_vel[:, right_ids].mean(dim=-1) * r
#     return torch.stack([v_left, v_right], dim=-1)
def wheel_encoder(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Returns a tensor of shape (N_envs, 2): [v_left, v_right] [m/s]
    by reading the joint velocities of the left & right throttle joints.
    """
    robot = env.scene[asset_cfg.name]               # Articulation
    # match the actual throttle joint names on your robot
    left_ids, _  = robot.find_joints(["left_.*_wheel_joint"])
    right_ids, _ = robot.find_joints(["right_.*_wheel_joint"])

    # joint_vel is (N_envs, n_joints)
    joint_vel = robot.data.joint_vel               # rad/s
    # convert from rad/s → m/s: v = ω * r
    r = 0.05  # wheel radius from your action cfg
    v_left  = joint_vel[:, left_ids].mean(dim=-1) * r
    v_right = joint_vel[:, right_ids].mean(dim=-1) * r
    return torch.stack([v_left, v_right], dim=-1)


def lidar_distances(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    """回傳 RayCaster 與 hit 點之間的距離 (m)。輸出 shape = (N_env, N_rays)."""
    sensor = env.scene[sensor_cfg.name]           # RayCaster 物件
    hits = sensor.data.ray_hits_w                 # (N, B, 3)
    origin = sensor.data.pos_w.unsqueeze(1)       # (N, 1, 3)
    dists = torch.norm(hits - origin, dim=-1)     # (N, B)
    return dists

GOAL = torch.tensor([5.0, 5.0])

def rel_heading(env, goal=GOAL):
    # bring goal to same device
    goal = goal.to(env.device)
    pos = mdp.root_pos_w(env)[..., :2]
    yaw = root_euler_xyz(env)[..., 2]            # ← use your imported function
    to_goal = torch.atan2(goal[1] - pos[...,1], goal[0] - pos[...,0])
    Δ = ((to_goal - yaw + torch.pi) % (2 * torch.pi)) - torch.pi
    return Δ.unsqueeze(-1)  # shape (B,1)

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

        rel_heading_term = ObsTerm(
            func=rel_heading,
            clip=(-torch.pi, torch.pi),
            noise=None
        )


        last_action_term = ObsTerm( # [m/s, (-1, 1)]
            func=mdp.last_action,
            clip=(-1., 1.), # TODO: get from ClipAction wrapper or action space
        )

        wheel_odom = ObsTerm(
            func=wheel_encoder,
            noise=Gnoise(std=0.01),
            clip=(-MAX_SPEED, MAX_SPEED),
        )
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()