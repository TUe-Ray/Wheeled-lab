import math
import torch

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

# ─────────────────────────────────────────────────────────
#  Task constants
GOAL        = (10.0, 0.0)
GOAL_RADIUS = 0.5
STOP_RADIUS = 5.0
MAX_SPEED   = 3.0
# ─────────────────────────────────────────────────────────
#  Scene
@configclass
class FlatPlaneCfg(TerrainImporterCfg):
    height = 0.0
    prim_path = "/World/ground"
    terrain_type = "plane"
    debug_vis = False

@configclass
class StraightLineSceneCfg(InteractiveSceneCfg):
    terrain = FlatPlaneCfg()
    robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
    )
# ─────────────────────────────────────────────────────────
#  Events
def reset_to_origin(env, env_ids, orientation):
    robots = env.scene["robot"]
    num    = len(env_ids)
    device = env.device
    pose   = torch.zeros((num, 7), dtype=torch.float32, device=device)
    pose[:, 3:7] = torch.tensor(orientation, device=device)
    robots.write_root_pose_to_sim(pose, env_ids=env_ids)

def decelerate_near_goal(env, env_ids, target, stop_radius):
    pos   = mdp.root_pos_w(env)[..., :2]
    tgt   = torch.tensor(target, device=pos.device)
    dist  = torch.norm(pos - tgt, dim=-1)
    ids   = [i for i in env_ids if dist[i] < stop_radius]
    if not ids:
        return
    acts = env.action_manager.current_actions[ids].clone()
    scale = (dist[ids] / stop_radius).unsqueeze(-1).clamp(0.0, 1.0)
    acts[:, 0:1] *= scale                 # throttle 在第 0 維
    env.action_manager.write_actions_to_sim(acts, env_ids=ids)

@configclass
class P2PEventsCfg:
    reset_to_origin = EventTerm(
        func=reset_to_origin,
        mode="reset",
        params={"orientation": None},     # 後面 env cfg 再填
    )
    decelerate = EventTerm(
        func=decelerate_near_goal,
        mode="post_step",
        params={"target": GOAL, "stop_radius": STOP_RADIUS},
    )
# ─────────────────────────────────────────────────────────
#  Rewards
def distance_to_goal(env, target=GOAL):
    pos = mdp.root_pos_w(env)[..., :2]
    tgt = torch.tensor(target, device=pos.device)
    return -torch.norm(pos - tgt, dim=-1)

@configclass
class P2PRewardsCfg:
    to_goal = RewTerm(func=distance_to_goal, weight=1.0, params={"target": GOAL})
# ─────────────────────────────────────────────────────────
#  Terminations
def reached_goal(env, target, r):
    pos = mdp.root_pos_w(env)[..., :2]
    tgt = torch.tensor(target, device=pos.device)
    return torch.norm(pos - tgt, dim=-1) < r

@configclass
class P2PTerminationsCfg:
    reached_goal = DoneTerm(func=reached_goal, params={"target": GOAL, "r": GOAL_RADIUS})
    time_out     = DoneTerm(func=mdp.time_out, time_out=True)
# ─────────────────────────────────────────────────────────
#  RL Env
@configclass
class MushrPoint2PointRLEnvCfg(ManagerBasedRLEnvCfg):
    seed        : int   = 42
    num_envs    : int   = 256
    env_spacing : float = 0.0

    observations : BlindObsCfg      = BlindObsCfg()
    actions      : MushrRWDActionCfg = MushrRWDActionCfg()
    rewards      : P2PRewardsCfg    = P2PRewardsCfg()
    events       : P2PEventsCfg     = P2PEventsCfg()
    terminations : P2PTerminationsCfg = P2PTerminationsCfg()
    curriculum   = None

    def __post_init__(self):
        super().__post_init__()

        # camera
        self.viewer.eye    = [10.0, 0.0, 20.0]
        self.viewer.lookat = [GOAL[0]/2.0, 0.0, 0.0]

        # sim timing
        self.sim.dt              = 0.005
        self.decimation          = 4
        self.sim.render_interval = 20
        self.episode_length_s    = 5

        # scale actions
        self.actions.throttle.scale = (MAX_SPEED,)
        self.actions.steer.scale    = (0.488,)

        # inject quaternion into reset event
        yaw = math.atan2(GOAL[1], GOAL[0])
        w = math.cos(yaw/2.0); x = y = 0.0; z = math.sin(yaw/2.0)
        self.events.reset_to_origin.params["orientation"] = (w, x, y, z)

        # scene
        self.scene = StraightLineSceneCfg(
            num_envs    = self.num_envs,
            env_spacing = self.env_spacing,
        )

# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = MushrPoint2PointRLEnvCfg()
    env = cfg.make()
    env.reset()
    print("Point-to-Point environment ready ✨")