import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.sim import SphereCfg, PreviewSurfaceCfg
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
##############################
###### COMMON CONSTANTS ######
##############################

CORNER_IN_RADIUS = 0.3        # For termination
CORNER_OUT_RADIUS = 2.0       # For termination
LINE_RADIUS = 0.8             # For spawning and reward
STRAIGHT = 0.8                # Shaping
SLIP_THRESHOLD = 0.55         # (rad) For reward
MAX_SPEED = 3.0               # (m/s) For action and reward

# somewhere at top‐level of your script
_prev_dist = None

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
    #robot: ArticulationCfg = OriginRobotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[5.0, 5.0, 0.0]),
        spawn=SphereCfg(
            radius=0.2,
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.0,0.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5 , 1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )
# LiDAR sensor
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",    #"{ENV_REGEX_NS}/Robot/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=["/World/envs/env_0/Obstacle1"],
        #mesh_prim_paths=["/World/ground"],

        
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-15.0,-15.0) ,
            horizontal_fov_range=(-180.0, 180.0),
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

@configclass
class DriftEventsCfg:
    # on startup

#    reset_root_state = EventTerm(
#        func=reset_root_state_along_track,
#        params={
#            "track_radius": LINE_RADIUS,
#            "track_straight_dist": STRAIGHT,
#            "num_points": 20,
#            "pos_noise": 0.5,
#            "yaw_noise": 1.0,
#            "asset_cfg": SceneEntityCfg("robot"),
#        },
#        mode="reset",
#    )


    reset_progress = EventTerm(
        func=reset_progress_tracker,
        mode="reset",
    )
    reset_root_state = EventTerm(
        func=reset_root_state_new,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos": [-2, -3, 0.0],    # ← your desired start-point A
            "rot": [0.0, 0.0, 0.0, 1.0], # no initial yaw
        },
        mode="reset",
    )
    clear_turn = EventTerm(
        func=clear_turn_buffers,
        mode="reset",
    )

    reset_progress = EventTerm(
        func=reset_dist_tracker,
        mode="reset",
    )

@configclass
class DriftEventsRandomCfg(DriftEventsCfg):

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.3, 0.5),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 20,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_link"),#body_names=".*wheel"),
            "make_consistent": True,
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*back.*throttle"]),  #joint_names=[".*_wheel_joint"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )

    # push_robots_hf = EventTerm( # High frequency small pushes
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(0.1, 0.4),
    #     params={
    #         "velocity_range":{
    #             "x": (-0.1, 0.1),
    #             "y": (-0.03, 0.03),
    #             "yaw": (-0.6, 0.6)
    #         },
    #     },
    # )

    # push_robots_lf = EventTerm( # Low frequency large pushes
    #     func=mdp.push_by_setting_velocity,
    #     mode="reset",
    #     interval_range_s=(0.8, 1.2),
    #     params={
    #         "velocity_range":{
    #             "yaw": (-1.5, 1.5)
    #         },
    #     },
    # )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),#body_names=["main_body"]),
            "mass_distribution_params": (0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )
######################
###### REWARDS #######
######################

_turn_buffers = None
_buf_params = (None, None)

def sustained_turn_reward(env, window_s: float = 10.0, tr: float = 0.5):
    global _turn_buffers, _buf_params

    dt = env.cfg.sim.dt * env.cfg.decimation
    window_steps= int(window_s)
    N = env.num_envs

    # re‐initialize if dims or window changed
    if _turn_buffers is None or _buf_params != (window_steps, N):
        _turn_buffers = [deque(maxlen=window_steps) for _ in range(N)]
        _buf_params = (window_steps, N)

    out = torch.zeros(N, device=env.device)
    w = mdp.base_ang_vel(env)[..., 2]  # yaw‐rate

    for i in range(N):
        buf = _turn_buffers[i]
        buf.append(float(w[i].cpu()))
        if len(buf) == buf.maxlen:
            avg_w = sum(buf) / buf.maxlen
            if abs(avg_w) > tr:
                out[i] = abs(avg_w)
    return out

# ────────────────────────────────────────────────────────────────────────────
# 2) REWARD FOR MOVING TOWARD GOAL
#    (signed projection: + when toward, – when away)
# ────────────────────────────────────────────────────────────────────────────

def signed_velocity_toward_goal(env, goal=torch.tensor([5.0, 5.0])):
    pos = mdp.root_pos_w(env)[..., :2]       # (B,2)
    vel = mdp.base_lin_vel(env)[..., :2]     # (B,2)

    to_goal = goal.to(env.device) - pos
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=-1)

    speed = torch.norm(vel, dim=-1)
    vel_norm = torch.nn.functional.normalize(vel + 1e-6, dim=-1)
    cosine = (vel_norm * to_goal_norm).sum(dim=-1)

    # positive if toward, negative if away
    return speed * cosine

# ────────────────────────────────────────────────────────────────────────────
# 3) PENALTY FOR MOVING AWAY OR STANDING STILL FAR FROM GOAL
#    a) Away‐movement penalty (only when projection < 0)
#    b) Distance penalty (scaled distance from goal)
# ────────────────────────────────────────────────────────────────────────────

def goal_reached_reward(env, goal=torch.tensor([5.0, 5.0]), threshold=0.3):
    pos = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal.to(env.device) - pos, dim=-1)
    #print("Goal reached reward:", torch.where(dist < threshold, 10.0, 0.0) )
    return torch.where(dist < threshold, 10.0, 0.0)

def away_movement_penalty(env, goal=torch.tensor([5.0, 5.0])):
    # clamp negative projections, zero otherwise
    proj = signed_velocity_toward_goal(env, goal=goal)
    return torch.clamp(-proj, min=0.0)

def distance_penalty(env, goal=torch.tensor([5.0, 5.0])):
    # straight‐line distance, penalize being far
    pos = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal.to(env.device) - pos, dim=-1)
    return -dist

@configclass
class TraverseABCfg:

   # encourage forward‐progress toward the goal
    # step_toward = RewTerm(
    #     func=signed_velocity_toward_goal,
    #     weight=20.0,
    # )

    # penalize any movement away
    away_penalty = RewTerm(
        func=away_movement_penalty,
        weight=30.0,
    )

    # penalize simply “parking” far from the goal
    dist_penalty = RewTerm(
        func=distance_penalty,
        weight=10.0,
    )

    # keep your alive and reach terms if you want
    alive = RewTerm(func=mdp.rewards.is_alive, weight=1.0)
    reach = RewTerm(func=goal_reached_reward, weight=50.0)

    # sustained turns (as before)
    sustained_turn = RewTerm(
        func=sustained_turn_reward,
        weight=905.0,
    )


########################
###### CURRICULUM ######
########################

@configclass
class DriftCurriculumCfg:
    # Every 20 eps reduce the alignment reward by 4, up to 5 times (20→0)
    decay_align = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "sustained_turn",
            "increase": -90,
            "episodes_per_increase": 4,
            "max_increases": 10,
        },
    )


##########################
###### TERMINATION #######
##########################

def cart_off_track(env, straight:float, corner_in_radius:float, corner_out_radius:float):
    out = torch.logical_or(
        off_track(env, straight, corner_out_radius) > 5,
        in_range(env, straight, corner_in_radius) > 5
    )
    return out

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

#    out_of_bounds = DoneTerm(
#        func=cart_off_track,
#        params={
#            "straight": STRAIGHT,
#            "corner_in_radius": CORNER_IN_RADIUS,
#            "corner_out_radius": CORNER_OUT_RADIUS
#        }
#    )

######################
###### RL ENV ########
######################

@configclass
class MushrDriftRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.

    # Basic Settings
    observations: BlindObsCfg = BlindObsCfg()
    # actions: MushrRWDActionCfg = MushrRWDActionCfg()
    actions: SkidSteerActionCfg = SkidSteerActionCfg()
    #actions: OriginActionCfg = OriginActionCfg()

    # MDP Settings
    rewards: TraverseABCfg = TraverseABCfg()
    events: DriftEventsCfg = DriftEventsRandomCfg()
    terminations: GoalNavTerminationsCfg = GoalNavTerminationsCfg()
    curriculum: DriftCurriculumCfg = DriftCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # viewer settings
        self.viewer.eye = [10., -10., 10.]
        self.viewer.lookat = [0.0, 0.0, 0.]

        self.sim.dt = 0.005  # 200 Hz
        self.decimation = 10  # 50 Hz
        self.sim.render_interval = 20 # 10 Hz
        self.episode_length_s = 10
        self.actions.throttle.scale = (MAX_SPEED, 6)

        self.observations.policy.enable_corruption = True

        # Scene settings
        self.scene = MushrDriftSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )

######################
###### PLAY ENV ######
######################

@configclass
class MushrDriftPlayEnvCfg(MushrDriftRLEnvCfg):
    """no terminations"""

    events: DriftEventsCfg = DriftEventsRandomCfg(
        reset_root_state = EventTerm(
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