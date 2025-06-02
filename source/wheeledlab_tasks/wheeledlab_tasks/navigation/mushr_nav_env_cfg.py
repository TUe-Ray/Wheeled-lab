import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from wheeledlab.envs.mdp import increase_reward_weight_over_time
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from wheeledlab_tasks.common import OriginActionCfg
from wheeledlab_assets import OriginRobotCfg
from wheeledlab_tasks.common.observations import NavObsCfg

from .mdp import  reset_root_state_new
from isaaclab.managers import TerminationTermCfg as DoneTerm


##############################
###### COMMON CONSTANTS ######
##############################

CORNER_IN_RADIUS = 0.3        # For termination
CORNER_OUT_RADIUS = 2.0       # For termination
LINE_RADIUS = 0.8             # For spawning and reward
STRAIGHT = 0.8                # Shaping
SLIP_THRESHOLD = 0.55         # (rad) For reward
MAX_SPEED = 1.5               # (m/s) For action and reward

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
    robot: ArticulationCfg = OriginRobotCfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],      # leave X,Y,Z at your desired spawn point
            # Quaternion [w, x, y, z] = [0,0,0,1] is a 180° rotation about Z
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,0.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0,1.0,3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )
# LiDAR sensor
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/main_body",
        update_period=1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=False,
        mesh_prim_paths=["/World/envs/env_0/Obstacle1"],
        #mesh_prim_paths=["/World/ground"],

        
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0,0) ,
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=12.0
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

    reset_root_state = EventTerm(
        func=reset_root_state_new,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos": [6.5, 0, 0.0],
            "rot": [0.0, 0.0, 0.0, 1.0], # no initial yaw
        },
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
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel"),
            "make_consistent": True,
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot",  joint_names=[".*_wheel_joint"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )

    push_robots_hf = EventTerm( # High frequency small pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.4),
        params={
            "velocity_range":{
                "x": (-0.1, 0.1),
                "y": (-0.03, 0.03),
                "yaw": (-0.3, 0.3)
            },
        },
    )

    push_robots_lf = EventTerm( # Low frequency large pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.8, 1.2),
        params={
            "velocity_range":{
                "yaw": (-0.6, 0.6)
            },
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["main_body"]),
            "mass_distribution_params": (0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )
######################
###### REWARDS #######
######################


def move_towards_goal(env, goal=torch.tensor([-4.0, 0.0]), scale=1.0, vel_weight=0.5):
    goal = torch.as_tensor(goal, dtype=torch.float32, device=env.device)  
    pos_xy = mdp.root_pos_w(env)[..., :2] 
    vec_to_goal = goal - pos_xy      
    dist = torch.norm(vec_to_goal, dim=-1) 
    dist_reward = torch.exp(-dist / scale)  
    vel_xy = mdp.root_lin_vel_w(env)[..., :2] 
    inv_dist = 1.0 / (dist.unsqueeze(-1) + 1e-6)  
    unit_to_goal = vec_to_goal * inv_dist 
    vel_proj = (vel_xy * unit_to_goal).sum(dim=-1)  
    return dist_reward + vel_weight * vel_proj

def goal_reached_reward(env, goal=[5.0,5.0], threshold=0.3):
    # turn the goal list into a tensor on the right device
    goal = torch.as_tensor(goal, device=env.device)
    pos  = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal - pos, dim=-1)
    return (dist < threshold).float()

########################
###### CURRICULUM ######
########################

@configclass
class DriftCurriculumCfg:

    more_slip = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "time_efficiency",
            "increase": 20.,
            "episodes_per_increase": 20,
            "max_increases": 10,
        }
    )

    more_term_pens = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "min_lidar_distance",
            "increase": -1.,
            "episodes_per_increase": 50,
            "max_increases": 5,
        }
    )

##########################
###### TERMINATION #######
##########################

def reached_goal(env, goal=[5.0,5.0], threshold=0.3):
    # turn the goal list into a tensor on the right device
    goal = torch.as_tensor(goal, device=env.device)
    pos  = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal - pos, dim=-1)
    # RETURN A BOOL TENSOR, NOT FLOAT
    return dist < threshold


def time_out_bool(env):
    return mdp.time_out(env).to(torch.bool)



@configclass
class GoalNavTerminationsCfg:

    time_out = DoneTerm(func=time_out_bool, time_out=True)
    
    success = DoneTerm(
        func=reached_goal,
        params={
            "goal": [-4.0, 0.0],
            "threshold": 0.5
        },
        time_out=False     
    )

def min_lidar_distance_penalty(env, threshold: float = 0.5):

    lidar = env.scene.sensors["ray_caster"]
    hits = lidar.data.ray_hits_w
    robot_pos = mdp.root_pos_w(env)[..., :2].unsqueeze(1)
    dists = torch.norm(hits[..., :2] - robot_pos, dim=-1)
    min_dist = dists.amin(dim=-1)
    return torch.where(
        min_dist < threshold,
        -1.0 + min_dist / threshold,
        torch.zeros_like(min_dist),
    )

def low_speed_penalty(env, low_speed_thresh: float = 0.3):
    vel_local = mdp.base_lin_vel(env)[..., 0]  
    return torch.where(
        vel_local.abs() < low_speed_thresh,  
        torch.ones_like(vel_local),
        torch.zeros_like(vel_local),
    )

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]

@configclass
class NavRewardsCfg:
    move_to_goal = RewTerm(
        func=move_towards_goal,
        weight=1.0,    
        params={
            "goal": [-4, 0],   
            "scale": 2.0,         
            "vel_weight": 0.3     
        },
    )

    reach_goal = RewTerm(
        func=goal_reached_reward,
        weight=100.0, 
        params={"goal": [-4, 0], "threshold": 0.5},
    )

    obstacle_penalty = RewTerm(
        func=min_lidar_distance_penalty,
        weight=20.0,   
        params={"threshold": 1.0},  
    )

    low_speed_penalty = RewTerm(
        func=low_speed_penalty,  
        weight=-5.0,             
        params={"low_speed_thresh": 0.3},
    )

######################
###### RL ENV ########
######################

@configclass
class MushrNavRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0

    observations: NavObsCfg = NavObsCfg()
    actions: OriginActionCfg = OriginActionCfg()

    # MDP Settings
    rewards: NavRewardsCfg      = NavRewardsCfg()
    #events: DriftEventsCfg = DriftEventsRandomCfg()
    events: DriftEventsCfg      = DriftEventsCfg()
    terminations: GoalNavTerminationsCfg = GoalNavTerminationsCfg()
    #curriculum: DriftCurriculumCfg = DriftCurriculumCfg()
    curriculum: None

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        print(" → action-cfg channels:", list(vars(self.actions).keys()))

        # viewer settings
        self.viewer.eye = [10., -10., 10.]
        self.viewer.lookat = [0.0, 0.0, 0.]

        self.sim.dt = 0.005  # 200 Hz --> 0.005
        self.decimation = 4  # 50 Hz
        self.sim.render_interval = 20 # 10 Hz
        self.episode_length_s = 20
        self.actions.throttle.scale = (MAX_SPEED, 3)

        self.observations.policy.enable_corruption = True

        # Scene settings
        self.scene = MushrDriftSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )

######################
###### PLAY ENV ######
######################

@configclass
class MushrNavPlayEnvCfg(MushrNavRLEnvCfg):
    """no terminations"""

    events: DriftEventsCfg = DriftEventsRandomCfg(
        reset_root_state = EventTerm(
            func=reset_root_state_new,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pos": [4, 0, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0], # no initial yaw
            },
            mode="reset",
        )
    )


    rewards: NavRewardsCfg = None
    terminations: GoalNavTerminationsCfg = None
    curriculum: DriftCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()