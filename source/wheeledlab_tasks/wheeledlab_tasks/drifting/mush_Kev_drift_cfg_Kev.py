import math
import torch
import inspect

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)

from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from wheeledlab_tasks.common import BlindObsCfg, MushrRWDActionCfg

##############################
###### COMMON CONSTANTS ######
##############################

GOAL         = (10.0, 0.0)   # target point A
GOAL_RADIUS  = 0.5           # meters for termination
STOP_RADIUS  = 5.0           # meters to begin deceleration
MAX_SPEED    = 3.0           # throttle scale

###################
###### SCENE ######
###################

@configclass
class FlatPlaneCfg(TerrainImporterCfg):
    height           = 0.0
    prim_path        = "/World/ground"
    terrain_type     = "plane"
    collision_group  = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode    = "multiply",
        restitution_combine_mode = "multiply",
        static_friction          = 1.1,
        dynamic_friction         = 1.0,
    )
    debug_vis        = False

@configclass
class StraightLineSceneCfg(InteractiveSceneCfg):
    terrain = FlatPlaneCfg()
    robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75,0.75,0.75), intensity=3000.0),
    )

    def __post_init__(self):
        super().__post_init__()
        yaw = math.atan2(GOAL[1], GOAL[0])
        w = math.cos(yaw/2.0)
        x = y = 0.0
        z = math.sin(yaw/2.0)
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.0),
            rot=(w, x, y, z),
        )

##########################
###### RESET FUNCTION ####
##########################

def reset_to_origin(env, env_ids, orientation):
    """
    Teleport the robot(s) back to start pose.
    """
    robots = env.scene["robot"]
    num = len(env_ids)
    device = env.device
    root_pose = torch.zeros((num, 7), dtype=torch.float32, device=device)
    quat = torch.tensor(orientation, dtype=torch.float32, device=device).unsqueeze(0)
    root_pose[:, 3:7] = quat
    robots.write_root_pose_to_sim(root_pose, env_ids=env_ids)

############################
###### DECELERATION #######
############################

def decelerate_near_goal(env, env_ids, target, stop_radius):
    """
    Gradually scale down throttle to zero as the vehicle approaches the goal.
    """
    # world positions of all agents
    pos = mdp.root_pos_w(env)[..., :2]
    tgt = torch.tensor(target, device=pos.device)
    dists = torch.norm(pos - tgt, dim=-1)  # (num_envs,)

    # identify which envs need deceleration
    to_decel = [i for i in env_ids if dists[i] < stop_radius]
    if not to_decel:
        return

    # fetch last actions for those envs
    # env.action_manager holds the actions before sim write
    last_acts = env.action_manager.current_actions  # Tensor shape (num_envs, action_dim)
    acts = last_acts[to_decel].clone()

    # compute linear scale: (d / stop_radius), clamped [0,1]
    scales = (dists[to_decel] / stop_radius).unsqueeze(-1)
    scales = torch.clamp(scales, 0.0, 1.0)


    # throttle is assumed to be the first action dimension
    acts[:, 0:1] *= scales

    # write modified actions back into sim for next step
    env.action_manager.write_actions_to_sim(acts, env_ids=to_decel)

####################
###### EVENTS ######
####################

@configclass
class StraightLineEventsCfg:
    reset_to_origin = EventTerm(
        func=reset_to_origin,
        mode="reset",
        params={"orientation": None},
    )
    decelerate = EventTerm(
        func=decelerate_near_goal,
        mode="post_step",
        params={"target": GOAL, "stop_radius": STOP_RADIUS},
    )

################
###### REWARDS##
################

def distance_to_goal(env, target=GOAL):
    pos = mdp.root_pos_w(env)[..., :2]
    tgt = torch.tensor(target, device=pos.device)
    return -torch.norm(pos - tgt, dim=-1)

@configclass
class StraightLineRewardsCfg:
    to_goal = RewTerm(
        func=distance_to_goal,
        weight=1.0,
        params={"target": GOAL},
    )

##########################
###### TERMINATION #######
##########################

def reached_goal_fn(env, target, r):
    pos = mdp.root_pos_w(env)[..., :2]
    tgt = torch.tensor(target, device=pos.device)
    return torch.norm(pos - tgt, dim=-1) < r

@configclass
class StraightLineTerminationsCfg:
    reached_goal = DoneTerm(
        func=reached_goal_fn,
        params={"target": GOAL, "r": GOAL_RADIUS},
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

######################
###### RL ENV ########
######################

@configclass
class MushrDriftRLEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.0

    observations: BlindObsCfg = BlindObsCfg()
    actions: MushrRWDActionCfg = MushrRWDActionCfg()
    rewards: StraightLineRewardsCfg      = StraightLineRewardsCfg()
    events:  StraightLineEventsCfg       = StraightLineEventsCfg()
    terminations: StraightLineTerminationsCfg = StraightLineTerminationsCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()

        # camera: top-down views
        self.viewer.eye    = [10.0, 0.0, 20.0]
        self.viewer.lookat = [10.0, 0.0, 0.0]

        # sim timing
        self.sim.dt              = 0.005
        self.decimation          = 10            #was 4
        self.sim.render_interval = 20
        self.episode_length_s    = 5

        # action scaling & obs noise
        self.actions.throttle_steer.scale = (MAX_SPEED, 0.488)
        self.observations.policy.enable_corruption = False

        # inject quaternion into reset event
        yaw = math.atan2(GOAL[1], GOAL[0])
        w = math.cos(yaw/2.0)
        x = y = 0.0
        z = math.sin(yaw/2.0)
        self.events.reset_to_origin.params["orientation"] = (w, x, y, z)

        # scene instantiation
        self.scene = StraightLineSceneCfg(
            num_envs   = self.num_envs,
            env_spacing= self.env_spacing,
        )

@configclass
class MushrDriftPlayEnvCfg(MushrDriftRLEnvCfg):
    rewards      = None
    terminations = None
    curriculum   = None

if __name__ == "__main__":
    cfg = MushrDriftRLEnvCfg()
    env = cfg.make()
    env.reset()
    print("Environment initialized and reset.")