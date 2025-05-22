from isaaclab.assets.articulation import ArticulationCfg

from . import WHEELEDLAB_ASSETS_DATA_DIR

OriginRobotCfg = ArticulationCfg(
    urdf_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/origin_v18.urdf",    # adjust if needed
    translation=[0.0, 0.0, 0.1],
    rotation=[1, 0, 0, 0],
    actuator_sets={
        "wheel_act": {
            "joint_names_expr": [".*wheel.*"],
            "type": "ideal_pd",
            "stiffness": 3.0, "damping": 60.0,
            "effort_limit": 20.0, "velocity_limit": 20.0
        }
    }
)