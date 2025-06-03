import torch
import isaacsim.core.utils.prims as prim_utils
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
from wheeledlab_tasks.common import ObsCfg, OriginActionCfg
from wheeledlab_assets import OriginRobotCfg

##############################
###### COMMON CONSTANTS ######
##############################

W_MAX = 1.0    # max |yaw rate| (rad/s)
V_MAX = 3.0    # max linear speed (m/s)
D_MAX = (10**2 + 10**2)**0.5  

###################
###### SCENE ######
###################

@configclass
class NavigationTerrainImporterCfg(TerrainImporterCfg):

    height = 0.0
    prim_path = "/World/ground"
    terrain_type="plane"
    collision_group = -1
    physics_material=sim_utils.RigidBodyMaterialCfg( 
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.1,
        dynamic_friction=1.0,
    )
    debug_vis=False
    visual_material = PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 1.0)   # RGB = white
    )
@configclass
class OriginOneNavigationSceneCfg(InteractiveSceneCfg):
    """Configuration for a OriginOne car Scene with racetrack terrain with no sensors"""

    terrain = NavigationTerrainImporterCfg()
    robot: ArticulationCfg = OriginRobotCfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        prim_path="{ENV_REGEX_NS}/obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0,2.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0,1.0,1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
        ),
    )

    obstacle2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacle2",
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
prev_dist = None

def reset_progress_tracker(env, env_ids):
    global _prev_dist
    _prev_dist = None
    return None  

_turn_buffers = None
def clear_turn_buffers(env, env_ids):
    global _turn_buffers
    if _turn_buffers is not None:
        if isinstance(env_ids, slice):
            ids = range(env.num_envs)
        elif hasattr(env_ids, "tolist"):
            ids = env_ids.tolist()
        else:
            ids = list(env_ids)
        for i in ids:
            _turn_buffers[i].clear()
    return torch.zeros(env.num_envs, device=env.device)

def reset_spin_timer(env, env_ids, duration: float = 1.0):
    """On reset, set every env’s spin timer to `duration` seconds."""
    global _spin_timers
    N = env.num_envs
    _spin_timers = torch.full((N,), duration, device=env.device)
    return torch.zeros(N, device=env.device)

# def spin_in_place(env, env_ids, max_w: float = 6.0):
#     """
#     Interval term: while each env’s timer > 0, command a random yaw velocity.
#     """
#     global _spin_timers
#     dt = env.cfg.sim.dt * env.cfg.decimation 
#     active = []
#     for i in env_ids.tolist():
#         if _spin_timers[i] > 0.0:
#             _spin_timers[i] -= dt
#             active.append(int(i))

#     if active:
#         active_ids = torch.tensor(active, device=env.device, dtype=torch.int64)
#         mdp.push_by_setting_velocity(
#             env,
#             env_ids=active_ids,
#             velocity_range={
#                 "x": (0.0, 0.0),
#                 "y": (0.0, 0.0),
#                 "yaw": (-max_w, max_w),
#             },
#         )

#     return torch.zeros(env.num_envs, device=env.device)

_step_counter: torch.Tensor | None = None
_time_left_paid: torch.Tensor | None = None

def reset_time_buffers(env, env_ids):
    """
    Called on every environment reset: zero out both our per‐env step counter
    and the “already gave finish‐bonus” flag.
    """
    global _step_counter, _time_left_paid
    N = env.num_envs

    if _step_counter is None or _step_counter.shape[0] != N:
        _step_counter   = torch.zeros(N, device=env.device, dtype=torch.int32)
        _time_left_paid = torch.zeros(N, device=env.device, dtype=torch.bool)
    else:
        _step_counter[env_ids]   = 0
        _time_left_paid[env_ids] = False

    return torch.zeros(env.num_envs, device=env.device)

@configclass
class NavigationEventsCfg:

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
                "yaw":   (-2, 2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_step_progress = EventTerm(
        func=reset_progress_tracker,
        mode="reset",
    )

    reset_time_buffers = EventTerm(
        func=reset_time_buffers,
        mode="reset",
    )


@configclass
class NavigationEventsRandomCfg(NavigationEventsCfg):

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.5, 1),
            "dynamic_friction_range": (0.5, 1),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]), 
            "damping_distribution_params": (10.0, 15.0),
            "operation": "abs",
        },
    )

    push_robots_hf = EventTerm( # High frequency small pushes
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.5, 2),
        params={
            "velocity_range":{
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "yaw": (-0.1, 0.1)
            },
        },
    )

    push_robots_lf = EventTerm( 
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
            "asset_cfg": SceneEntityCfg("robot",  body_names=["main_body"]),
            "mass_distribution_params": (0.05, 0.25),
            "operation": "add",
            "distribution": "uniform",
        },
    )

######################
###### REWARDS #######
######################

def forward_velocity_bonus(env) -> torch.Tensor:
    vel_world = mdp.base_lin_vel(env)[..., :2]  
    quat = mdp.root_quat_w(env)                 
    qw, qx, qy, qz = quat.unbind(-1)           
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )  
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  
    forward_speed = (vel_world * heading).sum(dim=-1)  
    forward_speed_clamped = forward_speed.clamp(min=0.0) 
    bonus = (forward_speed_clamped / V_MAX).clamp(max=1.0) 

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
    global _step_counter, _time_left_paid
    dt_per_step = env.cfg.sim.dt * env.cfg.decimation
    _step_counter += 1  

    elapsed_seconds = _step_counter.float() * dt_per_step  

    pos = mdp.root_pos_w(env)[..., :2]                              
    goal_tensor = torch.tensor(goal, device=env.device).unsqueeze(0) 
    dist = torch.norm(goal_tensor - pos, dim=-1)                     

    has_reached = dist < threshold             
    not_paid   = ~_time_left_paid              
    just_reached = has_reached & not_paid       

    out = torch.zeros(env.num_envs, device=env.device)  

    if just_reached.any():
        out[just_reached] = 1.0
        time_left = (env.cfg.episode_length_s - elapsed_seconds).clamp(min=0.0) 
        out[just_reached] += time_left[just_reached]
        _time_left_paid[just_reached] = True

    return out

def combined_lidar_velocity_penalty(    env,    min_dist: float = 0.5,    exponent: float = 2.0 , distance_weight: float = 1):
    lidar   = env.scene.sensors["ray_caster"]
    hits_w  = lidar.data.ray_hits_w        
    pos_xy = mdp.root_pos_w(env)[..., :2]    
    pos_world = pos_xy + env.scene.env_origins[:, :2] #Hits are in world coordinates but posxy not
    pos_world = pos_world.unsqueeze(1)     

    dist_all = torch.norm(hits_w[..., :2] - pos_world, dim=-1) 
    d_min, idx_min = dist_all.min(dim=-1)                  
    d_factor = ((min_dist - d_min) / min_dist).clamp(min=0.0, max=1.0) * distance_weight 


    batch_size = hits_w.shape[0]
    beam_indices = idx_min.view(batch_size, 1, 1).expand(batch_size, 1, 2)
    closest_hit_xy = torch.gather(hits_w[..., :2], dim=1, index=beam_indices).squeeze(1)  

    vec_to_obs = closest_hit_xy - pos_world.squeeze(1)   
    unit_to_obs = torch.zeros_like(vec_to_obs)           
    mask_close = (d_min < min_dist) & torch.isfinite(d_min) 
    unit_to_obs[mask_close] = torch.nn.functional.normalize(
        vec_to_obs[mask_close], dim=-1
    )

    vel_xy = mdp.base_lin_vel(env)[..., :2] 
    speed_toward = (vel_xy * unit_to_obs).sum(dim=-1)  
    speed_toward = speed_toward.clamp(min=0.0)         
    norm_speed = (speed_toward / V_MAX).clamp(max=1.0) * 1 - distance_weight

    combined = d_factor * norm_speed 
    penalty = - combined.pow(exponent)  
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
        weight=400,
        params={"min_dist": 1, "exponent": 2.0, "distance_weight": 0.4},
    )

    forward_bonus = RewTerm(
        func=forward_velocity_bonus,
        weight=10,   
    )

########################
###### CURRICULUM ######
########################

@configclass
class NavigationCurriculumCfg:


    decrease_dist_bonus = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "dist_bonus",
            "increase": -5,
            "episodes_per_increase": 15,
            "max_increases": 3,
        },
    )
    decrease_forward_bonus = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "forward_bonus",
            "increase": -3,
            "episodes_per_increase": 5,
            "max_increases": 3,
        },
    )

##########################
###### TERMINATION #######
##########################

def reached_goal(env, goal=[4.0, 4.0], thresh: float = 0.3):
    pos   = mdp.root_pos_w(env)[..., :2]          
    goal  = torch.tensor(goal, device=env.device).unsqueeze(0)  
    dist  = torch.norm(pos - goal, dim=-1)             
    reached = dist < thresh
    return reached

@configclass
class GoalNavTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    goal_reached = DoneTerm(
        func=reached_goal,
        params={
            "goal": [4.0, 4.0], 
            "thresh": 0.3
        }
    )
######################
###### RL ENV ########
######################

@configclass
class OriginOneNavigationRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 11.

    observations: ObsCfg = ObsCfg()
    actions: OriginActionCfg = OriginActionCfg()

    # MDP Settings
    rewards: TraverseABCfg = TraverseABCfg()
    events: NavigationEventsCfg = NavigationEventsRandomCfg()
    terminations: GoalNavTerminationsCfg = GoalNavTerminationsCfg()
    curriculum: NavigationCurriculumCfg = NavigationCurriculumCfg()

    def __post_init__(self):
                # Scene settings
        self.scene = OriginOneNavigationSceneCfg(
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
class OriginOneNavigationPlayEnvCfg(OriginOneNavigationRLEnvCfg):
    """no terminations"""

    events: NavigationEventsCfg = NavigationEventsRandomCfg(
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
    ))


    rewards: TraverseABCfg = None
    terminations: GoalNavTerminationsCfg = None
    curriculum: NavigationCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()