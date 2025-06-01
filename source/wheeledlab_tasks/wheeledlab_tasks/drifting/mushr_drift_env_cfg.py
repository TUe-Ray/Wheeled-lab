import torch
import numpy as np
import isaacsim.core.utils.prims as prim_utils
from itertools import product
import random
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg, MeshCuboidCfg, CollisionPropertiesCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)

from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from wheeledlab_tasks.common import BlindObsCfg, MushrRWDActionCfg, SkidSteerActionCfg, OriginActionCfg
from wheeledlab_assets import OriginRobotCfg
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from .mdp import reset_root_state_along_track, reset_root_state_new
from functools import partial
import math 
##############################
###### COMMON CONSTANTS ######
##############################


MAX_SPEED = 3.0               # (m/s) For action and reward
prev_dist = None

def reset_progress_tracker(env, env_ids):
    global _prev_dist
    _prev_dist = None
    return None   # EventTerms always expect a return, even if you don’t use it


from collections import deque


_turn_buffers = None
def clear_turn_buffers(env, env_ids):
    global _turn_buffers
    if _turn_buffers is not None:
        # normalize env_ids → list of ints
        if isinstance(env_ids, slice):
            ids = range(env.num_envs)
        elif hasattr(env_ids, "tolist"):
            ids = env_ids.tolist()
        else:
            ids = list(env_ids)
        for i in ids:
            _turn_buffers[i].clear()
    # return a dummy tensor so IsaacLab is happy
    return torch.zeros(env.num_envs, device=env.device)

def signed_velocity_toward_goal(env, goal=torch.tensor([5.0,5.0])):
    # 1) world-frame position & velocity
    pos = mdp.root_pos_w(env)[..., :2]          # (B,2)
    vel = mdp.base_lin_vel(env)[..., :2]        # (B,2)

    # 2) unit vector pointing from you → goal
    to_goal = goal.to(env.device) - pos         # (B,2)
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=-1)

    # 3) forward speed
    speed = torch.norm(vel, dim=-1)             # (B,)

    # 4) heading alignment (cosine of angle between vel & to_goal)
    vel_norm = torch.nn.functional.normalize(vel, dim=-1)
    cosine = (vel_norm * to_goal_norm).sum(dim=-1)  # in [–1,1]

    # 5) return signed‐projection:   speed × cos(θ)
    #    >0 if moving toward, <0 if moving away
    return speed * cosine


_prev_dists = None

def reset_dist_tracker(env, env_ids):
    global _prev_dists
    _prev_dists = None
    # no reward on reset
    return torch.zeros(env.num_envs, device=env.device)

def step_progress(env, goal=torch.tensor([5.0, 5.0])):
    global _prev_dists
    pos = mdp.root_pos_w(env)[..., :2]           # (B,2)
    dists = torch.norm(goal.to(env.device) - pos + 0.00001, dim=-1)  # (B,)

    if _prev_dists is None:
        # first call after reset → no progress
        prog = torch.zeros_like(dists)
    else:
        prog = _prev_dists - dists  # positive if we got closer

    _prev_dists = dists.clone()
    return prog

###################
###### SCENE ######
###################
@configclass
class DriftTerrainImporterCfg(TerrainImporterCfg):

    height = 0.0
    prim_path = "/World/ground"
    terrain_type="plane"
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg( # Material for carpet
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.1,
        dynamic_friction=1.0,
    )
    debug_vis=False

@configclass
class MushrDriftSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car Scene with racetrack terrain with no sensors"""

    terrain = DriftTerrainImporterCfg()
    robot: ArticulationCfg = OriginRobotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    #robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.0, 4.0, 0.0]),
        spawn=SphereCfg(radius=0.2,
                        visual_material=PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0))),
    )
    wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_north",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,  5.0, 0.75], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(10.0, 0.2, 1.5),  # length=10 in X, thickness=0.2 in Y, height=1.5 in Z
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )

    # ── South wall under /World/envs/env_i/wall_south ──
    wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_south",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -5.0, 0.75], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(10.0, 0.2, 1.5),  # length=10 in X, thickness=0.2 in Y, height=1.5 in Z
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )

   
    wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_west",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-5.0, 0.0, 0.75], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 10.0, 1.5),  # thickness=0.2 in X, length=10 in Y, height=1.5 in Z
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )

    # ── East wall under /World/envs/env_i/wall_east ──
    wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_east",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[ 5.0, 0.0, 0.75], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 10.0, 1.5),  # thickness=0.2 in X, length=10 in Y, height=1.5 in Z
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1.0,0.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0,1.0,1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )

    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0,0.0,0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=[            "/World/envs/env_.*/wall_west",
                                     "/World/envs/env_.*/wall_north",
                                     "/World/envs/env_.*/wall_south",
                                    "/World/envs/env_.*/wall_east",    
                                    "/World/envs/env_.*/obstacle1" ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-15.0,-15.0),
            horizontal_fov_range=(-180.0,180.0),
            horizontal_res=1.0
        ),
        debug_vis=False,
    )


    def __post_init__(self):
        """Post intialization."""
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0),
        )

#####################
###### EVENTS #######
#####################
# 1) global spin‐timer storage
_spin_timers = None

def reset_spin_timer(env, env_ids, duration: float = 1.0):
    """On reset, set every env’s spin timer to `duration` seconds."""
    global _spin_timers
    N = env.num_envs
    _spin_timers = torch.full((N,), duration, device=env.device)
    return torch.zeros(N, device=env.device)

def spin_in_place(env, env_ids, max_w: float = 6.0):
    """
    Interval term: while each env’s timer > 0, command a random yaw velocity.
    """
    global _spin_timers
    dt = env.cfg.sim.dt * env.cfg.decimation  # e.g. 0.005 * 10 = 0.05s

    # For each env in this batch, decrement its timer
    # and collect those still active
    active = []
    for i in env_ids.tolist():
        if _spin_timers[i] > 0.0:
            _spin_timers[i] -= dt
            active.append(int(i))

    # If any env still has timer > 0, push a random yaw to those
    if active:
        active_ids = torch.tensor(active, device=env.device, dtype=torch.int64)
        mdp.push_by_setting_velocity(
            env,
            env_ids=active_ids,
            velocity_range={
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (-max_w, max_w),
            },
        )

    return torch.zeros(env.num_envs, device=env.device)
  


# def randomize_goal(env, env_ids,
#                    area=(-5,5,-5,5),
#                    min_dist=1.0):
#     """
#     Move /World/envs/env_i/GoalMarker via its XFormPrim each reset.
#     """
#     stage = env.stage
#     # obstacle positions for distance check
#     obs_xform = XFormPrim(f"/World/envs/env_.*/Obstacle1", stage)
#     # we'll re-instantiate per env below
#     for i in env_ids.tolist():
#         # get robot’s freshly‐set start pos
#         rp = env.scene.articulations["robot"].data.root_pos_w[i,:2]
#         obs_pt = obs_xform.get_world_pose(env_ids=[i])[:, :2][0]

#         # rejection sample
#         while True:
#             gx = random.uniform(area[0], area[1])
#             gy = random.uniform(area[2], area[3])
#             pt = torch.tensor([gx,gy], device=env.device)
#             if ( (pt - rp    ).norm() >= min_dist
#                  and (pt - obs_pt ).norm() >= min_dist ):
#                 break

#         prim_path = f"/World/envs/env_{i}/GoalMarker"
#         xform_goal = XFormPrim(prim_path, stage)
#         xform_goal.set_world_pose(
#             position=[gx, gy, 0.0],
#             orientation=[1.0, 0.0, 0.0, 0.0],
#             env_ids=[i],
#         )
#     return torch.zeros(env.num_envs, device=env.device)

# # 3) RANDOMIZE ROBOT START
# def randomize_robot_start(env, env_ids,
#                           area=(-5,5,-5,5),
#                           min_dist=1.0):
#     """
#     Pick a start for each env that’s ≥min_dist from the obstacle and the goal.
#     Uses reset_root_state_new to actually teleport the car.
#     """
#     # get the current obstacle position in each env
#     obs = env.scene.extras["obstacle1"]
#     obs_pos = obs.get_world_pose(env_ids=env_ids)[:, :2]

#     goal = torch.tensor([5.0, 5.0], device=env.device)
#     for i in env_ids.tolist():
#         while True:
#             x = random.uniform(area[0], area[1])
#             y = random.uniform(area[2], area[3])
#             pt = torch.tensor([x,y], device=env.device)
#             if ( (pt - obs_pos[i]).norm() >= min_dist
#                  and (pt - goal    ).norm() >= min_dist ):
#                 break

#         # teleport it flat + identity yaw
#         reset_root_state_new(
#             env,
#             torch.tensor([i], device=env.device),
#             asset_cfg=SceneEntityCfg("robot"),
#             pos=[x, y, 0.0],
#             rot=[0.0, 0.0, 0.0, 1.0],
#         )
#     return torch.zeros(env.num_envs, device=env.device)




# 4) RANDOMIZE GOAL
@configclass
class DriftEventsCfg:
    # … your other reset terms …



    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Sample local x uniformly in [-5,5], same for y, and keep z=0
            "pose_range": {
                "x": (-4.5, 0.0),
                "y": (-4.5, 0.0),
                "z": (0.0, 0.0),
            },
            # No initial velocity
            "velocity_range": {
                "x":    (0.0, 0.0),
                "y":    (0.0, 0.0),
                "z":    (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch":(0.0, 0.0),
                "yaw":   (0.0, 0.0),
            },
            # Target the robot articulation
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    spin_in_place = EventTerm(
        func=spin_in_place,
        mode="interval",
        interval_range_s=(0.005 * 10, 0.005 * 10),  # every physics step
        params={"max_w": 6.0},
    )

    reset_step_progress = EventTerm(
        func=reset_progress_tracker,
        mode="reset",
    )

    # Track distance progress
    reset_dist_progress = EventTerm(
        func=reset_dist_tracker,
        mode="reset",
    )

    # )
    # randomize_robot_start = EventTerm(
    #     func=randomize_robot_start,
    #     mode="reset",
    #     params={"area":(-5,5,-5,5),"min_dist":2},
    # )
    # randomize_goal = EventTerm(
    #     func=randomize_goal,
    #     mode="reset",
    #     params={"area":(-5,5,-5,5),"min_dist":2},
    # )


@configclass
class DriftEventsRandomCfg(DriftEventsCfg):

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.6, 0.7),
            "dynamic_friction_range": (0.4, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel"), #body_names=".*wheel_link"),
            "make_consistent": True,
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]), #oint_names=[".*back.*throttle"]),  #
            "damping_distribution_params": (10.0, 15.0),
            "operation": "abs",
        },
    )

    push_robots_hf = EventTerm( # High frequency small pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1, 4),
        params={
            "velocity_range":{
                "x": (-0.1, 0.1),
                "y": (-0.03, 0.03),
                "yaw": (-0.6, 0.6)
            },
        },
    )

    push_robots_lf = EventTerm( # Low frequency large pushes
        func=mdp.push_by_setting_velocity,
        mode="startup",
        params={
            "velocity_range":{
                "yaw": (-2, 2)
            },
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot",  body_names=["main_body"]), #body_names=["base_link"]), #
            "mass_distribution_params": (0.1, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
######################
###### REWARDS #######
######################

W_MAX = 6.0    # max |yaw rate| (rad/s)
V_MAX = 3.0    # max linear speed (m/s)
D_MAX = (5.0**2 + 5.0**2)**0.5  # ≈7.07 m

_turn_buffers = None
_buf_params    = (None, None)

def sustained_turn_reward(env, window_s: float = 1.0, tr: float = 0.1):
    global _turn_buffers, _buf_params

    # how many steps in window_s seconds
    dt = env.cfg.sim.dt * 10
    window_steps = max(1, int(window_s / dt))
    N = env.num_envs

    # init or resize buffers
    if _turn_buffers is None or _buf_params != (window_steps, N):
        _turn_buffers = [deque(maxlen=window_steps) for _ in range(N)]
        _buf_params    = (window_steps, N)

    out = torch.zeros(N, device=env.device)
    w   = mdp.base_ang_vel(env)[..., 2].abs().clamp(max=W_MAX)

    for i in range(N):
        buf = _turn_buffers[i]
        buf.append(float(w[i].cpu()))
        if len(buf) == buf.maxlen:
            avg_w = sum(buf) / buf.maxlen
            if avg_w > tr:
                # normalize [tr..W_MAX] → [0..1]
                val = (avg_w - tr) / (W_MAX - tr)
                out[i] = val

    return out.clamp(0.0, 1.0)


def instant_turn_reward(env, scale: float = 1.0):
    w = mdp.base_ang_vel(env)[..., 2].abs().clamp(max=W_MAX)
    out = (w / W_MAX) * scale
    return out.clamp(0.0, 1.0)


def signed_velocity_toward_goal(env, goal=torch.tensor([5.0,5.0])):
    pos = mdp.root_pos_w(env)[..., :2]
    vel = mdp.base_lin_vel(env)[..., :2]

    to_goal      = goal.to(env.device) - pos
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=-1)

    speed    = torch.norm(vel, dim=-1).clamp(max=V_MAX)
    vel_norm = torch.nn.functional.normalize(vel + 1e-6, dim=-1)
    cosine   = (vel_norm * to_goal_norm).sum(dim=-1).clamp(-1.0,1.0)

    out = (speed * cosine) / V_MAX
    return out.clamp(-1.0, 1.0)


def away_movement_penalty(env, goal=torch.tensor([5.0,5.0])):
    proj = signed_velocity_toward_goal(env, goal=goal) * V_MAX
    raw  = torch.clamp(-proj, min=0.0)
    out  = raw / V_MAX
    return out.clamp(0.0, 1.0)


def distance_penalty(env, goal=torch.tensor([5.0,5.0])):
    pos  = mdp.root_pos_w(env)[...,:2]
    dist = torch.norm(goal.to(env.device)-pos, dim=-1).clamp(max=D_MAX)
    # normalize [0..D_MAX] → [1..0]
    return (1.0 - dist/D_MAX).clamp(0.0, 1.0)


def goal_reached_reward(env, goal=torch.tensor([5.0,5.0]), threshold=0.3):
    pos  = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal.to(env.device) - pos, dim=-1)
    out  = torch.where(dist < threshold, 1.0, 0.0)
    return out
V_TRANS_MAX = 3.0  # max linear speed (m/s)

def turn_in_place_reward(env):
    """
    Rewards yaw‐rate when the car is effectively not translating.
    Output is normalized to [0,1].
    """
    # 1) get absolute yaw‐rate (rad/s)
    w = mdp.base_ang_vel(env)[..., 2].abs()  # shape (B,)
    w = w.clamp(max=W_MAX)

    # 2) get translational speed (m/s)
    vel = mdp.base_lin_vel(env)[..., :2]  # shape (B,2)
    v = torch.norm(vel, dim=-1).clamp(max=V_TRANS_MAX)

    # 3) translation penalty factor in [0,1]
    #    = 1 when v=0 (pure spin), → 0 when v>=V_TRANS_MAX
    trans_factor = (1.0 - v / V_TRANS_MAX).clamp(0.0, 1.0)

    # 4) reward = (|w|/W_MAX) * trans_factor  ∈ [0,1]
    return (w / W_MAX) * trans_factor


def lidar_obstacle_penalty(env, min_dist: float = 0.3, exponent: float = 2.0):
    """
    Continuous obstacle‐proximity penalty:
      - If dist >= min_dist  →  0
      - If dist <  min_dist  →  ((min_dist - dist)/min_dist)**exponent
    Then we average over all beams to get a [0,1] signal per env.
    """
    # 1) pull out the front‐facing LiDAR hits
    lidar      = env.scene.sensors["ray_caster"]
    hits_w     = lidar.data.ray_hits_w              # (B, R, 3)
    positions  = mdp.root_pos_w(env)[..., :2].unsqueeze(1)  # (B, 1, 2)

    # 2) compute horizontal distance to each hit
    dist       = torch.norm(hits_w[..., :2] - positions, dim=-1)  # (B, R)
    # 3) how much inside the “safe” radius?
    delta      = (min_dist - dist).clamp(min=0.0)  # (B, R)

    # 4) normalize & raise to exponent
    penalty_beams = (delta / min_dist) ** exponent  # (B, R), in [0,1]

    # 5) average (or sum) across beams
    #    → mean gives you a [0,1] per‐env penalty
    return penalty_beams.mean(dim=-1)


def flip_penalty(env):
    """
    Penalize roll/pitch tipping:
      0 when flat, up to 1 when fully flipped.
    """
    # (w, x, y, z)
    quat = mdp.root_quat_w(env)          # shape (B,4)
    qw, qx, qy, qz = quat.unbind(-1)

    # compute roll & pitch
    num_r = 2*(qw*qx + qy*qz)
    den_r = 1 - 2*(qx*qx + qy*qy)
    roll  = torch.atan2(num_r, den_r)

    sinp  = (2*(qw*qy - qz*qx)).clamp(-1.0, 1.0)
    pitch = torch.asin(sinp)

    # normalize to [0,1]
    return -((roll.abs() + pitch.abs()) / math.pi).clamp(0.0, 1.0)

@configclass
class TraverseABCfg:

    step_toward = RewTerm(
        func=signed_velocity_toward_goal,
        weight=20.0,
    )

    # penalize any movement away
    away_penalty = RewTerm(
        func=away_movement_penalty,
        weight=0.2,
    )

    # penalize simply “parking” far from the goal
    dist_penalty = RewTerm(
        func=distance_penalty,
        weight=5,
    )

    # keep your alive and reach terms if you want
    alive = RewTerm(func=mdp.rewards.is_alive, weight=1.0)
    reach = RewTerm(func=goal_reached_reward, weight=500.0)

    # sustained turns (as before)
    # sustained_turn = RewTerm(
    #     func=sustained_turn_reward,
    #     weight=10,
    # )

    
    flip_penalty = RewTerm(
        func=flip_penalty,
        weight=500,    
    )
    
    obstacle_penalty = RewTerm(
        func=lidar_obstacle_penalty,
        weight=5.0,         # tweak as needed
        params={"min_dist": 0.3, "exponent": 2.0},
    )

########################
###### CURRICULUM ######
########################

@configclass
class DriftCurriculumCfg:
    # Every 20 eps reduce the alignment reward by 4, up to 5 times (

    decay_flip = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "flip_penalty",
            "increase": -100,
            "episodes_per_increase": 4,
            "max_increases": 5,
        },
    )




##########################
###### TERMINATION #######
##########################

def reached_goal(env, goal=[5.0, 5.0], thresh: float = 0.3):
    pos   = mdp.root_pos_w(env)[..., :2]               # B x 2
    goal  = torch.tensor(goal, device=env.device).unsqueeze(0)  # 1 x 2
    dist  = torch.norm(pos - goal, dim=-1)             # B]
    reached = dist < thresh
    if torch.any(reached):
        print("stopped with pos:", pos, "dist to goal", dist)
    return reached

@configclass
class GoalNavTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=reached_goal,
        params={
            "goal": [5.0, 5.0],  # Point B
            "thresh": 0.3
        }
    )



######################
###### RL ENV ########
######################

@configclass
class MushrDriftRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 11.

    # Basic Settings
    observations: BlindObsCfg = BlindObsCfg()
    # actions: MushrRWDActionCfg = MushrRWDActionCfg()
    #actions: SkidSteerActionCfg = SkidSteerActionCfg()
    actions: OriginActionCfg = OriginActionCfg()

    # MDP Settings
    rewards: TraverseABCfg = TraverseABCfg()
    events: DriftEventsCfg = DriftEventsRandomCfg()
    terminations: GoalNavTerminationsCfg = GoalNavTerminationsCfg()
    curriculum: DriftCurriculumCfg = DriftCurriculumCfg()

    def __post_init__(self):
                # Scene settings
        self.scene = MushrDriftSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )
        """Post initialization."""
        super().__post_init__()

        # viewer settings
        self.viewer.eye = [20., -20., 20.]
        self.viewer.lookat = [0.0, 0.0, 0.]

        self.sim.dt = 0.005  # 200 Hz
        self.decimation = 10  # 50 Hz
        self.sim.render_interval = 20 # 10 Hz
        self.episode_length_s = 15
        self.actions.throttle.scale = (MAX_SPEED, 6)

        self.observations.policy.enable_corruption = True



######################
###### PLAY ENV ######
######################

@configclass
class MushrDriftPlayEnvCfg(MushrDriftRLEnvCfg):
    """no terminations"""

    events: DriftEventsCfg = DriftEventsRandomCfg(
        reset_robot = EventTerm(
            func=reset_root_state_along_track,
            params={
                "dist_noise": 0.,
                "yaw_noise": 0.,
            },
            mode="reset",
        )
    )

    rewards: TraverseABCfg = None
    terminations: GoalNavTerminationsCfg = None
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()