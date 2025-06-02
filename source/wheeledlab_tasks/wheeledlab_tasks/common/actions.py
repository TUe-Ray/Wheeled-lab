from isaaclab.utils import configclass

from wheeledlab.envs.mdp import RCCar4WDActionCfg, RCCarRWDActionCfg

@configclass
class MushrRWDActionCfg:

    throttle_steer = RCCarRWDActionCfg(
        wheel_joint_names=[
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
        ],
        steering_joint_names=[
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ],
        base_length=0.325,
        base_width=0.2,
        wheel_radius=0.05,
        scale=(3.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )


@configclass
class Mushr4WDActionCfg:

    throttle_steer = RCCar4WDActionCfg(
        wheel_joint_names=[
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
            "front_left_wheel_throttle",
            "front_right_wheel_throttle",
        ],
        steering_joint_names=[
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ],
        base_length=0.325,
        base_width=0.2,
        wheel_radius=0.05,
        scale=(3.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )


from wheeledlab.envs.mdp import DifferentialDriveActionCfg

@configclass
class SkidSteerActionCfg:
    throttle = DifferentialDriveActionCfg(
        left_wheel_joint_names=[
            "front_left_wheel_throttle",
            "back_left_wheel_throttle",
        ],
        right_wheel_joint_names=[
            "front_right_wheel_throttle",
            "back_right_wheel_throttle",
        ],
        wheel_radius=0.12,
        wheel_base=0.5,   
        scale=(1.0, 1.0),         
        bounding_strategy="clip",
        no_reverse=False,
        asset_name="robot",
    )

@configclass
class OriginActionCfg:
    throttle = DifferentialDriveActionCfg(
        left_wheel_joint_names = ["left_front_wheel_joint", "left_rear_wheel_joint"],
        right_wheel_joint_names = ["right_front_wheel_joint", "right_rear_wheel_joint"],
        wheel_radius=0.12,
        wheel_base=0.5,       
        scale=(1.0, 1.0),   
        bounding_strategy="clip",
        no_reverse=False,
        asset_name="robot",
    )