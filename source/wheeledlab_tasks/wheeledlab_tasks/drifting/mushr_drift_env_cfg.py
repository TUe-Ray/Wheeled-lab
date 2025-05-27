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
from wheeledlab_tasks.common import BlindObsCfg, MushrRWDActionCfg, SkidSteerActionCfg, OriginActionCfg
from wheeledlab_assets import OriginRobotCfg
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from .mdp import reset_root_state_along_track, reset_root_state_new

##############################
###### COMMON CONSTANTS ######
##############################

CORNER_IN_RADIUS = 0.3        # For termination
CORNER_OUT_RADIUS = 2.0       # For termination
LINE_RADIUS = 0.8             # For spawning and reward
STRAIGHT = 0.8                # Shaping
SLIP_THRESHOLD = 0.55         # (rad) For reward
MAX_SPEED = 3.0               # (m/s) For action and reward

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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    obstacle1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[2.0,0.0,0.0], rot=[1,0,0,0]),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0,1.0,3.0),
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
        
        debug_vis=True,
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
    reset_root_state = EventTerm(
        func=reset_root_state_new,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos": [10, 10, 0.0],    # ← your desired start-point A
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
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),#body_names=["main_body"]),
            "mass_distribution_params": (0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )
######################
###### REWARDS #######
######################

def track_progress_rate(env):
    '''Estimate track progress by positive z-axis angular velocity around the environment'''
    asset : RigidObject = env.scene[SceneEntityCfg("robot").name]
    root_ang_vel = asset.data.root_link_ang_vel_w # this is different than the mdp one
    progress_rate = root_ang_vel[..., 2]
    return progress_rate

def vel_dist(env, speed_target: float=MAX_SPEED, offset: float=-MAX_SPEED**2):
    lin_vel = mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    speed_dist = (ground_speed - speed_target) ** 2 + offset
    return speed_dist # speed target

def cross_track_dist(env,
                    straight: float,
                    track_radius: float=(CORNER_IN_RADIUS + CORNER_OUT_RADIUS) / 2,
                    offset: float= -1.,
                    p: float=1.0):
    """Measures distance from a given radius on the track. Defaults
             to the middle of the track."""
    poses = mdp.root_pos_w(env)
    on_straights = torch.abs(poses[...,1]) < straight
    sq_ctd = torch.where(on_straights,
                torch.where(poses[...,0] > 0, # Straights
                    (poses[...,0] - track_radius)**2, # Quadrant 1
                    (poses[...,0] + track_radius)**2), # Quadrant 2
                torch.where(poses[...,1] > 0, # Corners
                    (torch.sqrt((poses[...,1] - straight)**2 + poses[...,0]**2) - track_radius)**2, # Positive y Turn
                    (torch.sqrt((poses[...,1] + straight)**2 + poses[...,0]**2) - track_radius)**2 # Negative y Turn
                )
    )
    ctd = torch.sqrt(sq_ctd) + offset

    return torch.pow(ctd, p)

def energy_through_turn(env, straight: float):
    poses = mdp.root_pos_w(env)
    speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    energy_through_turn = torch.where(torch.abs(poses[...,1]) > straight, speed**2, 0.)
    return energy_through_turn

def in_range(env, straight, corner_in_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) < corner_in_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 < corner_in_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 < corner_in_radius**2, 1, 0)))
    return penalty

def off_track(env, straight, corner_out_radius):
    poses = mdp.root_pos_w(env)
    penalty = torch.where(torch.abs(poses[...,1]) < straight,
                torch.where(torch.abs(poses[...,0]) > corner_out_radius, 1, 0),
                torch.where(poses[...,1] > 0,
                    torch.where((poses[...,1] - straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0),
                    torch.where((poses[...,1] + straight)**2 + poses[...,0]**2 > corner_out_radius**2, 1, 0)))
    return penalty

def side_slip(env, min_thresh: float, max_thresh: float, min_vel_x: float=0.5):
    vel = mdp.base_lin_vel(env)
    slip_angle = torch.abs(torch.atan2(vel[...,1], vel[...,0]))
    valid_angle = torch.where(torch.logical_or(
        torch.abs(vel[..., 0]) < min_vel_x, slip_angle > max_thresh),
        0.0, slip_angle
    )
    # Discount lateral vel from steering
    valid_angle = torch.where(valid_angle < min_thresh, 0.0, valid_angle)
    # Clamp unstable angles. Harder than zeroing for heavy, unstable vehicles
    # valid_angle = torch.clamp(valid_angle, max=max_thresh)
    return valid_angle

def turn_left_go_right(env, ang_vel_thresh: float=torch.pi/4):
    asset = env.scene[SceneEntityCfg("robot").name]
    steer_joints = asset.find_joints(".*_steer")[0]
    steer_joint_pos = mdp.joint_pos(env)[..., steer_joints].mean(dim=-1)
    ang_vel = mdp.base_ang_vel(env)[..., 2]
    ang_vel = torch.clamp(ang_vel, max=ang_vel_thresh, min=-ang_vel_thresh)
    tlgr = steer_joint_pos * ang_vel * -1.
    rew = torch.clamp(tlgr, min=0.)
    return rew

def move_towards_goal(env, goal=torch.tensor([5.0, 5.0]), scale=1.0):
    pos = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal.to(env.device) - pos, dim=-1)
    return torch.exp(-dist / scale)

def lidar_obstacle_penalty(env, min_dist=0.3):
    """
    Penalize if obstacle is closer than min_dist in front of robot.
    """
    lidar: RayCaster = env.scene.sensors["ray_caster"]
    hits = lidar.data.ray_hits_w  # (B x rays x 3)
    robot_pos = mdp.root_pos_w(env)[..., :2].unsqueeze(1)  # B x 1 x 2
    dist = torch.norm(hits[..., :2] - robot_pos, dim=-1)  # B x rays
    close_hits = (dist < min_dist).float()
    return -close_hits.sum(dim=-1)

def low_speed_penalty(env, low_speed_thresh: float=0.3):
    lin_speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    pen = torch.where(lin_speed < low_speed_thresh, 1., 0.)
    return pen

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]

def goal_direction_alignment(env, goal=torch.tensor([5.0, 5.0])):
    pos = mdp.root_pos_w(env)[..., :2]  # B x 2
    vel = mdp.base_lin_vel(env)[..., :2]  # B x 2
    to_goal = goal.to(env.device) - pos  # B x 2
    to_goal_norm = torch.nn.functional.normalize(to_goal, dim=-1)
    vel_norm = torch.nn.functional.normalize(vel, dim=-1)
    dot = (to_goal_norm * vel_norm).sum(dim=-1)  # B
    return dot  # +1 when aligned, -1 when opposite

def time_efficiency(env, goal=torch.tensor([5.0, 5.0]), reached_thresh=0.2):
    # current step number (scalar)
    step = max(int(env.common_step_counter), 1)
    pos = mdp.root_pos_w(env)[..., :2]
    dist_to_goal = torch.norm(goal.to(env.device) - pos, dim=-1)
    reached = dist_to_goal < reached_thresh
    # constant scalar divisor broadcasts to (B,)
    return reached.float() / float(step)


def min_lidar_distance_penalty(env, threshold=0.5):
    lidar: RayCaster = env.scene.sensors["ray_caster"]
    hits = lidar.data.ray_hits_w  # (B x rays x 3)
    robot_pos = mdp.root_pos_w(env)[..., :2].unsqueeze(1)  # B x 1 x 2
    dists = torch.norm(hits[..., :2] - robot_pos, dim=-1)
    min_dist = torch.min(dists, dim=-1)[0]  # B
    return torch.where(min_dist < threshold, -1.0 + min_dist / threshold, torch.zeros_like(min_dist))

def collision_penalty_contact_sensor(env, threshold=5.0):
    contact_sensor = env.scene.sensors["contact_sensor"]
    net_forces = contact_sensor.data.net_forces_w_history  # B x T x N x 3
    force_magnitudes = torch.norm(net_forces, dim=-1)  # B x T x N
    max_force = torch.max(force_magnitudes, dim=(1, 2))[0]
    return torch.where(max_force > threshold, -1.0, torch.zeros_like(max_force))

def smooth_velocity_change(env):
    vel = mdp.base_lin_vel(env)
    delta = vel - env.prev_vel  # requires env.prev_vel to be tracked manually
    return -torch.norm(delta, dim=-1)

def low_angular_velocity(env):
    ang_vel = mdp.base_ang_vel(env)
    return -torch.abs(ang_vel[..., 2])  # Penalize yaw

def goal_reached_reward(env, goal=torch.tensor([5.0, 5.0]), threshold=0.3):
    pos = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(goal.to(env.device) - pos, dim=-1)
    return torch.where(dist < threshold, 10.0, 0.0)


@configclass
class TraverseABCfg:
    goal_progress = RewTerm(
        func=move_towards_goal,
        weight=10.0,
    )

    obstacle_avoidance = RewTerm(
        func=lidar_obstacle_penalty,
        weight=5.0,
        params={"min_dist": 0.3},
    )

    alive = RewTerm(
        func=mdp.rewards.is_alive,
        weight=1.0,
    )

    timeout_penalty = RewTerm(
        func=mdp.rewards.is_terminated,
        weight=-50.0,
    )

    low_speed_penalty = RewTerm(
        func = low_speed_penalty,
        weight = 2
    )

    forward_vel = RewTerm(
        func = forward_vel,
        weight = 2,
    )
    align = RewTerm(func=goal_direction_alignment, weight=5.0)
    avoid = RewTerm(func=min_lidar_distance_penalty, weight=3.0)
    reach = RewTerm(func=goal_reached_reward, weight=50.0)
    time = RewTerm(func=time_efficiency, weight=10.0)
    stable = RewTerm(func=low_angular_velocity, weight=1.0)


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

#    more_tlgr = CurrTerm(
#        func=increase_reward_weight_over_time,
#        params={
#            "reward_term_name": "tlgr",
#            "increase": 10.,
#            "episodes_per_increase": 20,
#            "max_increases": 5,
#        }
#    )

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
        self.decimation = 4  # 50 Hz
        self.sim.render_interval = 20 # 10 Hz
        self.episode_length_s = 10
        self.actions.throttle.scale = (MAX_SPEED, 0.488)

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