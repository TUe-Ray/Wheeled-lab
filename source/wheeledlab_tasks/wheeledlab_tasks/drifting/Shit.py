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
    from wheeledlab_tasks.common import BlindObsCfg, MushrRWDActionCfg, SkidSteerActionCfg
    import math
    import random
    from .mdp import reset_root_state_along_track

    ##############################
    ###### COMMON CONSTANTS ######
    ##############################

    GOAL        = (10.0, 0.0)
    GOAL_RADIUS = 0.5
    STOP_RADIUS = 5.0
    MAX_SPEED   = 3.0

    ###################
    ###### SCENE ######
    ###################

    @configclass
    class GroundCfg(TerrainImporterCfg):

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

        terrain = GroundCfg()

        robot: ArticulationCfg = MUSHR_SUS_2WD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # lights
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        obstacle1 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Obstacle1",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0,0.0,0.0], rot=[1,0,0,0]),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.5,0.5,0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
            ),
        )
        # LiDAR sensor
        ray_caster = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
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

    def random_z_rotation_quaternion():
        theta = random.uniform(0, 2 * math.pi)  # 在 0 到 2π 間產生隨機角度（弧度）
        w = math.cos(theta / 2)
        x = 0.0
        y = 0.0
        z = math.sin(theta / 2)
        return (w, x, y, z)
    def reset_to_origin(env, env_ids, orientation):
        if orientation is None:
            orientation = random_z_rotation_quaternion()
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


    ######################
    ###### REWARDS #######
    ######################

    def neg_distance(env, target=GOAL):
        pos = mdp.root_pos_w(env)[..., :2]
        return -torch.norm(pos - torch.tensor(target, device=pos.device), dim=-1)

    @configclass
    class RewardsCfg:
        to_goal = RewTerm(func=neg_distance, weight=1.0, params={"target": GOAL})



    ########################
    ###### CURRICULUM ######
    ########################

    @configclass
    class CurriculumCfg:
        boost_goal = CurrTerm(
            func=increase_reward_weight_over_time,
            params={
                "reward_term_name": "to_goal",
                "increase": 0.2,
                "episodes_per_increase": 50,
                "max_increases": 10,
            }
        )



    ##########################
    ###### TERMINATION #######
    ##########################

    def reached(env, target, r):
        pos = mdp.root_pos_w(env)[..., :2]
        return torch.norm(pos - torch.tensor(target, device=pos.device), dim=-1) < r

    @configclass
    class TermsCfg:
        goal   = DoneTerm(func=reached, params={"target": GOAL, "r": GOAL_RADIUS})
        t_out  = DoneTerm(func=mdp.time_out, time_out=True)



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


        # MDP Settings
        rewards      : RewardsCfg      = RewardsCfg()
        events       : EventsCfg       = EventsCfg()
        terminations : TermsCfg        = TermsCfg()
        curriculum   : CurriculumCfg   = CurriculumCfg()   # ← 若不想用就設 None

        def __post_init__(self):
            """Post initialization."""
            super().__post_init__()

            # viewer settings
            self.viewer.eye = [4., -10., 6.]
            self.viewer.lookat = [0.0, 0.0, 0.]

            self.sim.dt = 0.005  # 200 Hz
            self.decimation = 4  # 50 Hz
            self.sim.render_interval = 20 # 10 Hz
            self.episode_length_s = 5
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
        """
        Play / Demo 環境：
        - 拿掉 Rewards、Terminations、Curriculum
        - 只保留 reset 事件，且關閉 decelerate，讓操控更直覺
        """

        # --- Events ----------------------------------------------------------
        # 1. 先用 EventsCfg(...) 的「可重載機制」複製一份原事件設定
        # 2. 把 decel 關掉（設 None），並重新指定 reset 事件
        #    → orientation 先留 None，稍後在 __post_init__ 注入
        events: EventsCfg = EventsCfg(
            reset = EventTerm(                     # 覆蓋原 reset
                func   = reset_to_origin,
                mode   = "reset",
                params = {"orientation": None},
            ),
            decel = None,                          # 關掉減速事件
        )

        # --- 其他 Manager 區塊全部取消 ---------------------------------------
        rewards      : RewardsCfg    = None
        terminations : TermsCfg      = None
        curriculum                     = None

        # ---------------------------------------------------------------------
        def __post_init__(self):
            """
            仍沿用父類別的視角、Sim 參數、Scene 等設定，
            但要把 quaternion 寫回剛剛留空的 reset.params["orientation"]。
            """
            super().__post_init__()

            # 把目標朝向轉成 quaternion (w,x,y,z)
            yaw = math.atan2(GOAL[1], GOAL[0])
            quat = (math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2))

            # 寫回到 events.reset.params
            self.events.reset.params["orientation"] = quat

            # 依需要還可以再加攝影機、渲染等個別設定
            # self.viewer.eye    = [...]
            # self.viewer.lookat = [...]

