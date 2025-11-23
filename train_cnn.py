"""
FIXED CNN Training - No Atari Wrappers, Headless Compatible

Keywords: headless training, custom environment, no display, server training
"""
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Set headless mode for pygame (CRITICAL for servers)
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from breakout_env_cnn import BreakoutEnv


def make_env():
    """
    Create environment WITHOUT Atari wrappers
    (Our custom environment doesn't need them)
    """
    env = BreakoutEnv()
    env = Monitor(env)
    return env


if __name__ == "__main__":
    print("=" * 80)
    print("FIXED CNN TRAINING - HEADLESS & CUSTOM ENV COMPATIBLE")
    print("=" * 80)
    print("\nFixes Applied:")
    print("  ✓ SDL_VIDEODRIVER=dummy - Headless pygame rendering")
    print("  ✓ Removed Atari wrappers - Not needed for custom env")
    print("  ✓ VecNormalize - Reward normalization")
    print("  ✓ Lower learning rate - Stable updates")
    print("  ✓ Larger batches - Better gradients")
    print("=" * 80)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Frame stacking (4 frames for temporal info)
    env = VecFrameStack(env, n_stack=4)
    
    # Transpose for PyTorch: (H,W,C) -> (C,H,W)
    env = VecTransposeImage(env)
    
    # ⭐ CRITICAL: Reward normalization
    env = VecNormalize(
        env,
        training=True,
        norm_obs=False,  # Don't normalize images (already 0-255)
        norm_reward=True,  # NORMALIZE REWARDS (critical!)
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    )
    
    print(f"\n✓ Environment created")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  → (4, 84, 84) = 4 stacked grayscale frames")
    
    # Create directories
    os.makedirs("checkpoints_cnn_stable", exist_ok=True)
    os.makedirs("best_model_cnn_stable", exist_ok=True)
    os.makedirs("logs_cnn_stable", exist_ok=True)
    os.makedirs("tensorboard_cnn_stable", exist_ok=True)
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_cnn_stable/",
        name_prefix="ppo_breakout_cnn",
        verbose=1,
    )

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecNormalize(
        eval_env,
        training=False,  # Don't update during eval
        norm_obs=False,
        norm_reward=False,  # Get true rewards
        clip_obs=10.0,
        gamma=0.99,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_cnn_stable/",
        log_path="./logs_cnn_stable/",
        eval_freq=25_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    # ⭐ STABLE CNN CONFIGURATION
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        
        # Learning rate - REDUCED for stability
        learning_rate=1e-4,
        
        # Buffer and batch - INCREASED for stability
        n_steps=512,
        batch_size=512,
        
        # PPO parameters
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        
        # Entropy
        ent_coef=0.01,
        
        # Value function
        vf_coef=0.5,
        
        # Gradient clipping - STRICT
        max_grad_norm=0.5,
        
        # CNN architecture
        policy_kwargs=dict(
            features_extractor_kwargs=dict(
                features_dim=512
            ),
            normalize_images=False,
        ),
        
        tensorboard_log="./tensorboard_cnn_stable/",
        seed=42,
    )

    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Device:            {model.device}")
    print(f"Learning Rate:     1e-4 (reduced)")
    print(f"Batch Size:        512 (increased)")
    print(f"Buffer Size:       512 steps")
    print(f"Reward Norm:       ✓ ENABLED")
    print(f"Total Timesteps:   2,000,000")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("TRAINING METRICS TO WATCH")
    print("=" * 80)
    print("Good signs:")
    print("  - ep_rew_mean: Steadily increasing (20 → 50 → 100 → 200)")
    print("  - value_loss: Decreasing (45 → 10 → 3)")
    print("  - explained_variance: High and stable (0.7-0.9)")
    print("\nBad signs (if you see these, STOP):")
    print("  - ep_rew_mean: Decreasing (collapse!)")
    print("  - value_loss: Exploding (>50)")
    print("  - explained_variance: Dropping (<0.5)")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print("Monitor with: tensorboard --logdir ./tensorboard_cnn_stable/")
    print("=" * 80)
    print()

    # Train
    try:
        model.learn(
            total_timesteps=2_000_000,
            callback=callback,
            progress_bar=True,
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        
        # Save model and normalization
        model.save("ppo_breakout_cnn_stable_final")
        env.save("vec_normalize_cnn_stable.pkl")
        
        print("✓ Model saved: ppo_breakout_cnn_stable_final.zip")
        print("✓ Normalization saved: vec_normalize_cnn_stable.pkl")
        print("\nTest with:")
        print("  python test_cnn_stable.py --model ppo_breakout_cnn_stable_final")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        model.save("ppo_breakout_cnn_stable_interrupted")
        env.save("vec_normalize_cnn_stable_interrupted.pkl")
        print("✓ Model saved: ppo_breakout_cnn_stable_interrupted.zip")
        print("✓ Normalization saved: vec_normalize_cnn_stable_interrupted.pkl")
    
    finally:
        env.close()
        eval_env.close()


# WHAT CHANGED FROM PREVIOUS VERSION:
#
# 1. Added: os.environ['SDL_VIDEODRIVER'] = 'dummy'
#    → Enables headless pygame (no display needed)
#
# 2. Removed: NoopResetEnv and MaxAndSkipEnv
#    → These are Atari-specific wrappers
#    → Our custom environment doesn't need them
#    → They expect get_action_meanings() method
#
# 3. Added: Directory creation
#    → Ensures all paths exist before training
#
# WHY ATARI WRAPPERS AREN'T NEEDED:
#
# NoopResetEnv: Adds random no-ops at episode start
#   → Purpose: Reduce determinism in Atari games
#   → Our env: Already has randomized starts (80% easy)
#   → Not needed!
#
# MaxAndSkipEnv: Repeats actions for N frames, returns max
#   → Purpose: Speed up Atari games, handle flickering sprites
#   → Our env: Pygame doesn't flicker, already fast
#   → Not needed!
#
# These wrappers are useful for REAL Atari ROMs, but our custom
# environment already handles everything they provide.