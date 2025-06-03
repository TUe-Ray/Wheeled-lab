"""
Fast On-Policy runner that honours LogConfig cadence flags.
Put this next to your current runner and import it instead of the stock one.
"""

from __future__ import annotations
import os, time, threading, warnings, torch, numpy as np
from collections import deque
from rsl_rl import runners

# ───────── optional: silence tqdm experimental warning ───────
try:
    from tqdm import TqdmExperimentalWarning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None


class OnPolicyRunner(runners.OnPolicyRunner):
    """
    Same public interface as rsl-rl’s runner, but:
        • no per-step CPU sync (episode stats batched on GPU)
        • logging / checkpoints controlled by LogConfig
        • async torch.save
    """

    # ------------------------------------------------------------
    def __init__(self, env, agent_cfg, log_cfg, device="cuda:0"):
        super().__init__(env, agent_cfg, log_cfg.run_log_dir, device)

        # store the whole config so we can read cadence values later
        self.log_cfg      = log_cfg
        self.no_log       = log_cfg.no_log or log_cfg.test_mode
        self.no_wandb     = log_cfg.no_wandb or log_cfg.test_mode
        self.no_ckpt      = log_cfg.no_checkpoints or log_cfg.test_mode

        self.logger_type  = None if self.no_wandb else "wandb"
        self.log_every    = max(1, log_cfg.log_every)
        self.ckpt_every   = max(1, log_cfg.checkpoint_every)

    # ------------------------------------------------------------
    def learn(self, num_learning_iterations: int,
              init_at_random_ep_len: bool = False) -> None:

        # 1) optional WandB writer
        if not self.no_log and self.logger_type == "wandb":
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter
            self.writer = WandbSummaryWriter(
                log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
            )
            self.writer.log_config(self.env.cfg, self.cfg,
                                   self.alg_cfg, self.policy_cfg)

        # 2) reset env (unchanged)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        obs, extras   = self.env.get_observations()
        critic_obs    = extras["observations"].get("critic", obs)
        self.train_mode()

        # 3) per-env GPU accumulators
        ep_return = torch.zeros(self.env.num_envs, device=self.device)
        ep_len    = torch.zeros(self.env.num_envs, device=self.device)

        # host rolling buffers
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)

        start_iter = self.current_learning_iteration
        end_iter   = start_iter + num_learning_iterations

        # progress bar only when *not* piping through WandB
        use_bar = (tqdm is not None) and not self.no_wandb
        iterator = tqdm(range(start_iter, end_iter), disable=not use_bar)

        # ─────────────────────────────────────────────────────────
        for it in iterator:

            # ───── rollout ────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, rewards, dones, infos = self.env.step(actions)

                    # GPU-side episode stats
                    ep_return += rewards.squeeze(-1)
                    ep_len    += 1
                    done_mask  = dones.squeeze(-1)

                    if done_mask.any():
                        finished_rew = ep_return[done_mask]
                        finished_len = ep_len[done_mask]
                        # reset counters in-place
                        ep_return[done_mask] = 0.0
                        ep_len[done_mask]    = 0
                        # one batched host hop
                        rewbuffer.extend(finished_rew.cpu().tolist())
                        lenbuffer.extend(finished_len.cpu().tolist())

                    critic_obs = (
                        self.critic_obs_normalizer(infos["observations"]["critic"])
                        if "critic" in infos["observations"] else obs
                    )

            # compute returns & update
            self.alg.compute_returns(critic_obs)
            loss_dict = self.alg.update()

            # ───── logging (scalar) ───────────────────────────
            if (not self.no_log and not self.no_wandb and
                    it % self.log_every == 0):
                scalar = {
                    "iter": it,
                    "reward_mean": float(np.mean(rewbuffer)) if rewbuffer else 0.0,
                    "len_mean":    float(np.mean(lenbuffer)) if lenbuffer else 0.0,
                    **{f"loss/{k}": v for k, v in loss_dict.items()},
                }
                self.writer.log_scalars(scalar)
                rewbuffer.clear(); lenbuffer.clear()

            # ───── checkpoints (async) ────────────────────────
            if (not self.no_ckpt) and it % self.ckpt_every == 0:
                ckpt_path = os.path.join(
                    self.log_cfg.model_save_path, f"model_{it}.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                threading.Thread(
                    target=self.save, args=(ckpt_path,), daemon=True
                ).start()

            # optional console progress (rare)
            if it % max(1, self.log_every * 5) == 0:
                print(f"[iter {it:6d}] reward μ="
                      f"{float(np.mean(rewbuffer)) if rewbuffer else 0:.1f}")

            self.current_learning_iteration = it
