import torch
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

GOAL = torch.tensor([4.0, 4.0])
MAX_SPEED = 3.0
WHEEL_RADIUS = 0.12

def wheel_encoder(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Returns a tensor of shape (N_envs, 2): [v_left, v_right] [m/s]
    by reading the joint velocities of the left & right throttle joints.
    """
    robot = env.scene[asset_cfg.name]               
    left_ids, _  = robot.find_joints(["left_.*_wheel_joint"])
    right_ids, _ = robot.find_joints(["right_.*_wheel_joint"])

    joint_vel = robot.data.joint_vel               
    v_left  = joint_vel[:, left_ids].mean(dim=-1) * WHEEL_RADIUS
    v_right = joint_vel[:, right_ids].mean(dim=-1) * WHEEL_RADIUS
    return torch.stack([v_left, v_right], dim=-1)


def lidar_distances(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("ray_caster")):
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w
    origin = sensor.data.pos_w.unsqueeze(1)
    dists = torch.norm(hits - origin, dim=-1) 

    B, N = dists.shape

    def pool_region(start_deg, end_deg, num_bins):
        start_idx = int((start_deg + 180) / 360 * N)
        end_idx = int((end_deg + 180) / 360 * N)
        region = dists[:, start_idx:end_idx]
        step = max(1, region.shape[1] // num_bins)
        pooled = torch.stack([
            region[:, i * step:(i + 1) * step].min(dim=-1).values
            for i in range(num_bins)
        ], dim=-1)
        return pooled  


    front = pool_region(-45, 45, 10)          # 90° front → 6 bins
    rear = pool_region(150, 180, 3)          # rear-right
    rear = torch.cat([rear, pool_region(-180, -150, 3)], dim=-1)  # rear-left (2 bins total)
    left = pool_region(90, 150, 3)           # left side → 2 bins
    right = pool_region(-150, -90, 3)        # right side → 2 bins

    return torch.cat([front, left, right, rear], dim=-1)  # (B, 12)


def to_goal_vector(env, goal: torch.Tensor = GOAL) -> torch.Tensor:
    pos_xy = mdp.root_pos_w(env)[..., :2]         

    quat   = mdp.root_quat_w(env)    
    qw, qx, qy, qz = quat.unbind(-1)
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )                                               

    delta = goal.to(env.device) - pos_xy          
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    x_rel =  cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    y_rel = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]

    return torch.stack((x_rel, y_rel), dim=-1)  
   
@configclass
class ObsCfg:
    """Default observation configuration"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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
            noise =Gnoise(mean =0.,std=0.2),
            clip=(0.0,50.0)

        )

        to_goal_vector_term = ObsTerm(
            func=to_goal_vector,
            clip=(-torch.pi, torch.pi),
            noise=Gnoise(std = 0.3),
        )

        last_action_term = ObsTerm( 
            func=mdp.last_action,
            clip=(-1., 1.), # TODO: 
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()