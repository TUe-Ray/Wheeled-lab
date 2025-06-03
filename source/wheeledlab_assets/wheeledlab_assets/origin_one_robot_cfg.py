from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg

WHEEL_RADIUS     = 0.1175
MAX_WHEEL_SPEED  = 2.0               # [m s⁻¹]
FRICTION_STATIC  = 1.5               # tyres & ground
FRICTION_DYNAMIC = 1.2
ANG_DAMPING      = 3.0               # N m s / (rad s⁻¹)
MAX_ANG_VEL      = 30.0              # [rad s⁻¹] ≈ 5 rps

wheel_material = sim_utils.RigidBodyMaterialCfg(
    static_friction   = FRICTION_STATIC,
    dynamic_friction  = FRICTION_DYNAMIC,
    restitution       = 0.0,
    friction_combine_mode = "max",   # “stickiest wins”
)

OriginRobotCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "..", "data", "Robots", "origin_v18.urdf")
        ),
        joint_drive=None,
        fix_base=False,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,
        root_link_name="main_body",
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01, rest_offset=0.0
        ),
    ),
    rigid_props= sim_utils.RigidBodyPropertiesCfg(
        angular_damping       = ANG_DAMPING,
        max_angular_velocity  = MAX_ANG_VEL,
    ),

    actuators={
        "wheel_act": IdealPDActuatorCfg(
            joint_names_expr = [".*_wheel_joint"],
            stiffness        = 50,
            damping          = 5,
            effort_limit     = 10.0,
            velocity_limit   = MAX_WHEEL_SPEED / WHEEL_RADIUS,
        )
    },

    bodies={
        ".*_wheel$": dict(                         # any link ending with “_wheel”
            physics_material  = wheel_material,
            lateral_friction_scale  = 3.0,         # ⟂ to rolling direction
            rolling_friction_scale  = 0.5,         # //  (optional)
            spinning_friction_scale = 0.5,
        ),
    },
)



# import os

# from isaaclab.assets import ArticulationCfg
# import isaaclab.sim as sim_utils
# from isaaclab.actuators import IdealPDActuatorCfg


# WHEEL_RADIUS = 0.1175
# MAX_WHEEL_SPEED = 2  

# OriginRobotCfg = ArticulationCfg(
#     prim_path="{ENV_REGEX_NS}/Robot",
#     spawn=sim_utils.UrdfFileCfg(
#         asset_path=os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "data", "Robots", "origin_v18.urdf")
# ),
#         joint_drive=None,
#         fix_base=False,
#         merge_fixed_joints=True,
#         convert_mimic_joints_to_normal_joints=True,
#         root_link_name="main_body",
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.1),
#         joint_pos={},  # optionally set joints like "left_front_wheel_joint": 0.0
#     ),
#     actuators={
#         "wheel_act": IdealPDActuatorCfg(
#             joint_names_expr=[".*_wheel_joint"],
#             stiffness=50,
#             damping=5,
#             effort_limit=10.0,
#             velocity_limit=MAX_WHEEL_SPEED / WHEEL_RADIUS,
#         )
#     },
# )