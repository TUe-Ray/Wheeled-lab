# ─────────────────────────────────────────────────────────
import math, torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs    import ManagerBasedRLEnvCfg
from isaaclab.scene   import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils   import configclass
from isaaclab.assets  import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
from wheeledlab_assets import MUSHR_SUS_2WD_CFG
from wheeledlab_tasks.common import BlindObsCfg, SkidSteerActionCfg
# ─────────────────────────────────────────────────────────
GOAL        = (10.0, 0.0)
GOAL_RADIUS = 0.5
STOP_RADIUS = 5.0
MAX_SPEED   = 3.0
# ─────────────────────────────────────────────────────────
@configclass
class GroundCfg(TerrainImporterCfg):
    prim_path = "/World/ground"
    terrain_type = "plane"
    height = 0.0
    debug_vis = False

@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = GroundCfg()
    robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0),
    )
# ─────────────────────────────────────────────────────────
def reset_to_origin(env, env_ids, orientation):
    robots = env.scene["robot"]
    pose = torch.zeros((len(env_ids), 7),
                       dtype=torch.float32, device=env.device)
    pose[:, 3:7] = torch.tensor(orientation, device=env.device)
    robots.write_root_pose_to_sim(pose, env_ids=env_ids)

def decelerate_near_goal(env, env_ids, target, stop_radius):
    pos  = mdp.root_pos_w(env)[..., :2]
    dist = torch.norm(pos - torch.tensor(target, device=pos.device), dim=-1)
    ids  = [i for i in env_ids if dist[i] < stop_radius]
    if not ids:
        return
    acts  = env.action_manager.current_actions[ids].clone()  # (k,2)
    scale = (dist[ids] / stop_radius).unsqueeze(-1).clamp(0., 1.)
    acts[:, 0:1] *= scale          # v
    acts[:, 1:2] *= torch.sqrt(scale)  # w
    env.action_manager.write_actions_to_sim(acts, env_ids=ids)

@configclass
class EventsCfg:
    reset = EventTerm(func=reset_to_origin, mode="reset",
                      params={"orientation": None})
    decel = EventTerm(func=decelerate_near_goal, mode="post_step",
                      params={"target": GOAL, "stop_radius": STOP_RADIUS})
# ─────────────────────────────────────────────────────────
def neg_distance(env, target=GOAL):
    pos = mdp.root_pos_w(env)[..., :2]
    return -torch.norm(pos - torch.tensor(target, device=pos.device), dim=-1)

@configclass
class RewardsCfg:
    to_goal = RewTerm(func=neg_distance, weight=1.0, params={"target": GOAL})
# ─────────────────────────────────────────────────────────
def reached(env, target, r):
    pos = mdp.root_pos_w(env)[..., :2]
    return torch.norm(pos - torch.tensor(target, device=pos.device), dim=-1) < r

@configclass
class TermsCfg:
    goal   = DoneTerm(func=reached, params={"target": GOAL, "r": GOAL_RADIUS})
    t_out  = DoneTerm(func=mdp.time_out, time_out=True)
# ─────────────────────────────────────────────────────────
@configclass
class P2PRLEnvCfg(ManagerBasedRLEnvCfg):
    seed, num_envs, env_spacing = 42, 128, 0.0

    observations : BlindObsCfg        = BlindObsCfg()
    actions      : SkidSteerActionCfg = SkidSteerActionCfg()
    rewards      : RewardsCfg         = RewardsCfg()
    events       : EventsCfg          = EventsCfg()
    terminations : TermsCfg           = TermsCfg()
    curriculum   = None

    def __post_init__(self):
        super().__post_init__()
        # camera
        self.viewer.eye, self.viewer.lookat = [10,0,18], [GOAL[0]/2,0,0]
        # sim timing
        self.sim.dt, self.decimation = 0.005, 4
        self.episode_length_s        = 5
        # action scales
        self.actions.v.scale = (MAX_SPEED,)
        self.actions.w.scale = (0.488,)
        # inject orientation
        yaw = math.atan2(GOAL[1], GOAL[0])
        w,x,y,z = math.cos(yaw/2),0,0,math.sin(yaw/2)
        self.events.reset.params["orientation"] = (w,x,y,z)
        # scene
        self.scene = SceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = P2PRLEnvCfg()
    env = cfg.make()
    env.reset()
    print("Skid-Steer P2P env ready 🎉")