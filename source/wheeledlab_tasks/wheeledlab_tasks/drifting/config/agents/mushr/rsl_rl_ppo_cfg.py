from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class MushrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 512
    max_iterations = 150
    save_interval = 50
    experiment_name = "ppo_mushr"

    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.5,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
