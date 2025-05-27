
###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Train an RL Agent in WheeledLab.")
parser.add_argument('-r', "--run-config-name", type=str, default="RSS_DRIFT_CONFIG", help="Run in headless mode.")
#parser.add_argument('-r', "--run-config-name", type=str, default="RSS_LEV_CONFIG", help="Run in headless mode.")
print("at least you got here")

simulation_app, args_cli = startup(parser=parser)
print(" past startup")
#######################
###### END SETUP ######
#######################

import gymnasium as gym
import os
import torch

# Set the W&B API Key
os.environ["WANDB_API_KEY"] = "cce39c00168b08312b009079d9efa5b8e64f7806"
os.environ["WANDB_USERNAME"] = "jleliveld" 

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml, dump_pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl import WHEELEDLAB_RL_LOGS_DIR

from wheeledlab_rl.utils import (
    OnPolicyRunner as ModifiedRslRunner,
    CustomRecordVideo,
    hydra_run_config,
    ClipAction,
)

def check_for_nan_in_obs(obs):
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"[NaN] Found in obs[{k}]")
    elif isinstance(obs, (list, tuple)):
        for i, v in enumerate(obs):
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"[NaN] Found in obs[{i}]")
    elif isinstance(obs, torch.Tensor):
        if torch.isnan(obs).any():
            print("[NaN] Found in obs tensor")
    else:
        print("Unknown obs type:", type(obs))

@hydra_run_config(run_config_name=args_cli.run_config_name)
def main(run_cfg: RunConfig): # TODO: Add SB3 config support

    #################
    #### LOGGING ####
    #################

    ##### Aliasing Configs #####
    env_cfg = run_cfg.env
    agent_cfg = run_cfg.agent
    train_cfg = run_cfg.train
    log_cfg = train_cfg.log
    env_setup = run_cfg.env_setup

    if not log_cfg.no_wandb:
        import wandb
        wandb.login(key="41bb1124b81ea3364629063efbf3964a07de7a6f")
        run = wandb.init(
            project=log_cfg.wandb_project,
        )
        log_cfg.run_name = wandb.run.name

    if not os.path.exists(log_cfg.model_save_path):
        os.makedirs(log_cfg.model_save_path)

    ## UPDATE CONFIGS WANDB ##
    if not log_cfg.no_wandb:
        wandb.config.update(run_cfg.to_dict())

    # Save the config file
    if not log_cfg.no_log:
        dump_yaml(os.path.join(log_cfg.run_log_dir, "run_config.yaml"), run_cfg)
        dump_pickle(os.path.join(log_cfg.run_log_dir, "run_config.pkl"), run_cfg)

    ############################
    #### CREATE ENVIRONMENT ####
    ############################

    env = gym.make(env_setup.task_name, cfg=env_cfg, render_mode="rgb_array" if log_cfg.video else None)

    ####### INSTANTIATE ENV #######
    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env)

    # Wrap the environment in recorder
    if log_cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_cfg.run_log_dir, "videos"),
            "step_trigger": lambda step: step % log_cfg.video_interval == 0,
            "video_length": log_cfg.video_length,
            "disable_logger": True,
            "enable_wandb": not log_cfg.no_wandb,
            "video_resolution": log_cfg.video_resolution,
            "video_crf": log_cfg.video_crf,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)

    # TODO: add back support for SB3
    env = RslRlVecEnvWrapper(env)

    # obs = env.reset()
    # check_for_nan_in_obs(obs)

    runner = ModifiedRslRunner(env, agent_cfg.to_dict(), log_cfg, device=train_cfg.device)

    ##### LOAD EXISTING RUN? #####

    if train_cfg.load_run is not None:
        chkpt = "model_"
        if train_cfg.load_run_checkpoint > 0:
            chkpt = f"{chkpt}{train_cfg.load_run_checkpoint}"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(WHEELEDLAB_RL_LOGS_DIR, run_dir=train_cfg.load_run,
                                        other_dirs=["models"], checkpoint=f"{chkpt}.*")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    #################
    ##### TRAIN #####
    #################

    env.seed(agent_cfg.seed)
    env.common_step_counter = train_cfg.set_env_step # For continuing curriculums
    runner.learn(num_learning_iterations=train_cfg.num_iterations)

    print("runner.alg.policy is a", type(runner.alg.policy))
    print("=== runner.alg.policy attributes ===")
    print(dir(runner.alg.policy))

    print("=== Actor model architecture ===")
    print(runner.alg.policy.actor)
    actor_model = runner.alg.policy.actor
    torch.save(actor_model, os.path.join(log_cfg.model_save_path, "full_actor_model.pt"))
    print("[INFO] Saved full actor model.")
    
    if not log_cfg.no_wandb:
        run.finish()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()