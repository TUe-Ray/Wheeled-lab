import torch
from isaaclab.managers import ActionTerm

class DifferentialDriveAction(ActionTerm):
    """Differential Drive (Skid Steer) action term for IsaacLab."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, 2), device=self.device)
        self._processed_actions = torch.zeros((env.num_envs, 2), device=self.device)
        self._asset = self._env.scene[cfg.asset_name]

        # Get joint indices
        self._left_ids, _ = self._asset.find_joints(cfg.left_wheel_joint_names)
        self._right_ids, _ = self._asset.find_joints(cfg.right_wheel_joint_names)

        # Params
        self.wheel_radius = torch.tensor(cfg.wheel_radius, device=self.device)
        self.wheel_base = torch.tensor(cfg.wheel_base, device=self.device)

        # Scaling
        self._scale = torch.tensor(cfg.scale, device=self.device)
        self._bounding_strategy = cfg.bounding_strategy

        # Action placeholders
        
    @property
    def action_dim(self) -> int:
        return 2  # linear velocity v, angular velocity w

    @property
    def raw_actions(self):
        return self._raw_actions
    
    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        if self._bounding_strategy == "clip":
            self._processed_actions = torch.clip(actions, -1.0, 1.0) * self._scale
        elif self._bounding_strategy == "tanh":
            self._processed_actions = torch.tanh(actions) * self._scale
        else:
            self._processed_actions = actions * self._scale

        if self.cfg.no_reverse:
            self._processed_actions[:, 0] = torch.clamp(self._processed_actions[:, 0], min=0.0)

    def apply_actions(self):
        v = self._processed_actions[:, 0]  # linear velocity
        w = self._processed_actions[:, 1]  # angular velocity

        # Convert to individual wheel velocities
        v_l = (v - 0.5 * self.wheel_base * w) / self.wheel_radius
        v_r = (v + 0.5 * self.wheel_base * w) / self.wheel_radius

        # Apply to all left/right wheels
        left_targets = v_l.unsqueeze(-1).repeat(1, len(self._left_ids))
        right_targets = v_r.unsqueeze(-1).repeat(1, len(self._right_ids))

        self._asset.set_joint_velocity_target(left_targets, joint_ids=self._left_ids)
        self._asset.set_joint_velocity_target(right_targets, joint_ids=self._right_ids)