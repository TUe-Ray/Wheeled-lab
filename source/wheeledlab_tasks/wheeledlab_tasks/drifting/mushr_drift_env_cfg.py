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

from wheeledlab.envs.mdp import increase_reward_weight_over_time, anneal_min_dist_and_weight
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

W_MAX = 2.0    # max |yaw rate| (rad/s)
V_MAX = 3.0    # max linear speed (m/s)
D_MAX = (10**2 + 10**2)**0.5  
prev_dist = None

def reset_progress_tracker(env, env_ids):
    global _prev_dist
    _prev_dist = None
    return None   # EventTerms always expect a return, even if you don’t use it



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

_prev_dists = None

def reset_dist_tracker(env, env_ids):
    global _prev_dists
    _prev_dists = None
    # no reward on reset
    return torch.zeros(env.num_envs, device=env.device)

def step_progress(env, goal=torch.tensor([4.0, 4.0])):
    global _prev_dists
    pos = mdp.root_pos_w(env)[..., :2]           
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[4.0, 4.0, 0.0]),
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0,2.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0,1.0,1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )

    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0,0.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5,1.5),
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
                                    "/World/envs/env_.*/obstacle1",
                                     "/World/envs/env_.*/obstacle2" ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0, 0),
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
  

# 4) RANDOMIZE GOAL

@configclass
class DriftEventsCfg:

    reset_spin_timer = EventTerm(
        func=reset_spin_timer,
        mode="reset",
        params={"duration": 0.25},
    )

    reset_robot = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-4.5, 0.0),
                "y": (-4.5, 0.0),
                "z": (0.0, 0.0),
            },
            "velocity_range": {
                "x":    (0.0, 0.0),
                "y":    (0.0, 0.0),
                "z":    (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch":(0.0, 0.0),
                "yaw":   (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    spin_in_place = EventTerm(
        func=spin_in_place,
        mode="interval",
        interval_range_s=(0.005 * 10, 0.005 * 10),  
        params={"max_w": 6.0},
    )

    reset_step_progress = EventTerm(
        func=reset_progress_tracker,
        mode="reset",
    )

    reset_dist_progress = EventTerm(
        func=reset_dist_tracker,
        mode="reset",
    )


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



_turn_buffers = None

def forward_velocity_bonus(env) -> torch.Tensor:
    """
    A small reward ∈ [0,1] proportional to how fast the car is driving forward.
    - If the car’s forward‐component of velocity ≥ max_speed, reward = 1.0.
    - If it’s going backward or zero, reward = 0.0.
    - Otherwise reward = (forward_speed / max_speed).

    Args:
        env: the ManagerBasedEnv instance
        max_speed: speed (m/s) at which a full bonus of 1.0 is given.

    Returns:
        Tensor of shape (B,) in [0,1], one entry per environment.
    """
    # 1) Get world‐space linear velocity (B,3); we only care about XY:
    vel_world = mdp.base_lin_vel(env)[..., :2]  # (B, 2)

    # 2) Extract robot’s yaw from its world quaternion:
    quat = mdp.root_quat_w(env)                 # (B, 4) = (qw, qx, qy, qz)
    qw, qx, qy, qz = quat.unbind(-1)            # each is (B,)

    # 3) Compute yaw angle (heading in XY‐plane):
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )  # (B,)

    # 4) Build a unit‐vector “forward direction” in world‐XY:
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (B, 2)

    # 5) Project vel_world onto heading to get forward‐speed:
    forward_speed = (vel_world * heading).sum(dim=-1)  # (B,)

    # 6) Clamp negative (no bonus for going backwards):
    forward_speed_clamped = forward_speed.clamp(min=0.0)  # (B,)

    # 7) Normalize so that speeds ≥ max_speed give a 1.0 bonus:
    bonus = (forward_speed_clamped / V_MAX).clamp(max=1.0)  # (B,) ∈ [0,1]

    return bonus

def signed_velocity_toward_goal(env, goal=torch.tensor([4.0,4.0])):
    pos = mdp.root_pos_w(env)[..., :2]
    vel = mdp.base_lin_vel(env)[..., :2]

    to_goal      = goal.to(env.device) - pos
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=-1)
    speed    = torch.norm(vel, dim=-1).clamp(max=V_MAX)
    vel_norm = torch.nn.functional.normalize(vel + 1e-6, dim=-1)
    cosine   = (vel_norm * to_goal_norm).sum(dim=-1).clamp(-1.0,1.0)

    out = (speed * cosine) / V_MAX
    return out.clamp(-1.0, 1.0)


def distance_bonus(env, goal=torch.tensor([4.0,4.0])):
    pos  = mdp.root_pos_w(env)[...,:2]
    dist = torch.norm(goal.to(env.device)-pos, dim=-1).clamp(max=D_MAX)
    norm = (1.0 - dist/D_MAX).clamp(0.0, 1.0)
    return norm**2


def goal_reached_reward(env, goal=torch.tensor([4.0, 4.0]), threshold: float = 0.3):
    """
    At the step when the agent’s position comes within `threshold` of `goal`,
    pay out a one‐time bonus equal to (episode_length_s − elapsed_time). Otherwise 0.

    Args:
        env:            ManagerBasedEnv instance.
        goal:           2D goal point.
        threshold:      Distance threshold for “goal reached.”

    Returns:
        Tensor of shape (B,) containing 0 everywhere except at the moment of first goal‐reach,
        where it returns “seconds left in episode.”
    """

    # 1) Compute current XY‐distance to goal for each env
    pos = mdp.root_pos_w(env)[..., :2]                          # (B, 2)
    goal_t = goal.to(env.device).unsqueeze(0)                    # (1, 2)
    dist = torch.norm(goal_t - pos, dim=-1)                      # (B,)

    # 2) Mask of “just reached” (distance < threshold)
    reached = dist < threshold                                   # (B,), dtype=bool

    # 3) Figure out how much time has elapsed so far in each environment
    dt      = env.cfg.sim.dt * env.cfg.decimation                 # seconds per step
    elapsed = env.progress_buf.float() * dt                       # (B,), elapsed seconds

    # 4) Compute “seconds left” in the episode
    leftover = (env.episode_length_s - elapsed).clamp(min=0.0)     # (B,)

    # 5) Zero‐initialize bonus, then fill in only at the moment of goal‐reach
    bonus = torch.zeros_like(leftover, device=env.device)         # (B,)
    bonus[reached] = leftover[reached]

    return bonus

def combined_lidar_velocity_penalty(    env,    min_dist: float = 0.5,    exponent: float = 2.0 , distance_weight: float = 1):
    """
    Single‐pass LiDAR penalty that penalizes “closeness” and “driving toward”
    under a single exponent.  Returns a (B,) tensor of <= 0.

    Steps:
      1) Read ray_hits_w from the LIDAR: shape (B, R, 3).
      2) Compute d_min = closest‐beam‐distance in XY.
      3) Build d_factor = clamp((min_dist - d_min)/min_dist, 0, 1).  → distance term.
      4) Find the (x,y) of the closest hit; compute robot→hit direction.
      5) Project base_lin_vel onto that direction → speed_toward, clamp≥0, normalize by V_MAX.
      6) Multiply d_factor × (norm_speed_toward) → combined_factor, and raise to “exponent”.
      7) Return – combined_factor**exponent (so that when d_min ≥ min_dist OR speed_toward ≤ 0, penalty=0).

    Over time, you could reduce the “weight” on the distance factor (e.g. anneal min_dist) so the agent
    shifts focus to simply not ramming at high speed.
    """
    # 1) grab all hits:
    lidar   = env.scene.sensors["ray_caster"]
    hits_w  = lidar.data.ray_hits_w         # shape (B, R, 3)

    # 2) robot’s XY in world coordinates:
    pos_xy = mdp.root_pos_w(env)[..., :2]    # (B, 2)
    # if your scene uses env_origins to separate each arena, add them:
    pos_world = pos_xy + env.scene.env_origins[:, :2]  # (B, 2)
    pos_world = pos_world.unsqueeze(1)      # (B, 1, 2) for broadcasting

    # 3) compute horizontal distances from robot to every beam:
    dist_all = torch.norm(hits_w[..., :2] - pos_world, dim=-1)  # (B, R)
    d_min, idx_min = dist_all.min(dim=-1)                       # both are (B,)

    # 4) distance factor: (min_dist − d_min)/min_dist, clamped to [0,1]:
    d_factor = ((min_dist - d_min) / min_dist).clamp(min=0.0, max=1.0) * distance_weight # (B,)

    # 5) locate the XY of the closest hit (for direction)
    batch_size = hits_w.shape[0]
    beam_indices = idx_min.view(batch_size, 1, 1).expand(batch_size, 1, 2)
    closest_hit_xy = torch.gather(hits_w[..., :2], dim=1, index=beam_indices).squeeze(1)  # (B, 2)

    # 6) build unit‐vector from robot→closest hit, but only where d_min < min_dist:
    vec_to_obs = closest_hit_xy - pos_world.squeeze(1)   # (B, 2)
    unit_to_obs = torch.zeros_like(vec_to_obs)            # (B, 2)
    mask_close = (d_min < min_dist) & torch.isfinite(d_min)  # (B,)
    unit_to_obs[mask_close] = torch.nn.functional.normalize(
        vec_to_obs[mask_close], dim=-1
    )

    # 7) project robot’s XY velocity onto that direction:
    vel_xy = mdp.base_lin_vel(env)[..., :2]  # (B, 2)
    speed_toward = (vel_xy * unit_to_obs).sum(dim=-1)  # (B,)
    speed_toward = speed_toward.clamp(min=0.0)         # no penalty for “driving away”
    norm_speed = (speed_toward / V_MAX).clamp(max=1.0) * 1 - distance_weight# (B,)

    # 8) combined factor = d_factor * norm_speed, → zero if d_min ≥ min_dist or speed_toward ≤ 0
    combined = d_factor * norm_speed  # (B,)

    # 9) raise to exponent and negate:
    penalty = - combined.pow(exponent)  # (B,) ≤ 0
    return penalty


@configclass
class TraverseABCfg:

    step_toward = RewTerm(
        func=signed_velocity_toward_goal,
        weight= 20.0,
    )

    dist_bonus = RewTerm(
        func=distance_bonus ,
        weight=20,
    )

    reach = RewTerm(func=goal_reached_reward, weight=500.0)
    
    obstacle_velocity_penalty = RewTerm(
        func=combined_lidar_velocity_penalty,
        weight=50.0,
        params={"min_dist": 1.25, "exponent": 2.0, "distance_weight": 0.1},
    )

    forward_bonus = RewTerm(
        func=forward_velocity_bonus,
        weight=1.0,   # scale this bonus as you like
        params={"max_speed": 3.0},
    )

########################
###### CURRICULUM ######
########################

@configclass
class DriftCurriculumCfg:


    increase_penalty = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "dist_bonus",
            "increase": -5,
            "episodes_per_increase": 20,
            "max_increases": 3,
        },
    )


    anneal_obstacle = CurrTerm(
        func=anneal_min_dist_and_weight,
        params={
            "term_name":     "obstacle_velocity_penalty",
            "start_min_dist": 1.25,
            "end_min_dist":   0.75,
            "start_weight":   1,
            "end_weight":     0.1,
            "episodes_per_step": 10,
            "max_steps":     100,
        },
    )

##########################
###### TERMINATION #######
##########################

def reached_goal(env, goal=[4.0, 4.0], thresh: float = 0.3):
    pos   = mdp.root_pos_w(env)[..., :2]          
    goal  = torch.tensor(goal, device=env.device).unsqueeze(0)  # 1 x 2
    dist  = torch.norm(pos - goal, dim=-1)             # B]
    reached = dist < thresh
    # if torch.any(reached):
    #     print("stopped with pos:", pos, "dist to goal", dist)
    return reached

@configclass
class GoalNavTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=reached_goal,
        params={
            "goal": [4.0, 4.0],  # Point B
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
        self.actions.throttle.scale = (V_MAX, W_MAX)

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