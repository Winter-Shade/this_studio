import pygame
import numpy as np
from stable_baselines3 import PPO
from breakout_env_cnn import BreakoutEnv
import argparse
from collections import deque


class FrameStackWrapper:
    """
    Simple frame stacking wrapper without using gymnasium's FrameStack.
    Stacks the last n frames together.
    """
    def __init__(self, env, n_frames=4):
        self.env = env
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame buffer with the initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs[:, :, 0])  # Remove channel dimension
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs[:, :, 0])  # Remove channel dimension
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        # Stack frames: (n_frames, height, width)
        return np.stack(list(self.frames), axis=0)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space


def make_visual_env():
    """
    Create environment for visualization with human rendering.
    
    The BreakoutEnv outputs 84x84x1 grayscale observations.
    We stack 4 frames to match the training configuration.
    """
    env = BreakoutEnv(render_mode="human")
    env = FrameStackWrapper(env, n_frames=4)
    return env


def visualize(model_path, episodes=5, deterministic=True):
    """
    Visualize a trained PPO agent playing Breakout.
    
    Args:
        model_path: Path to the trained model (.zip file)
        episodes: Number of episodes to run
        deterministic: If True, agent uses deterministic policy
    """
    # Load trained model
    print(f"\n🤖 Loading model: {model_path}")
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create wrapped environment
    env = make_visual_env()
    print("✅ Environment ready")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")

    # Initialize pygame font for HUD
    pygame.init()
    font = pygame.font.Font(None, 28)

    print(f"\n🎮 Starting {episodes} episode(s)...\n")

    episode_rewards = []
    episode_steps = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"Episode {ep} starting...")

        while not done:
            # Handle pygame events (allows window to be closed)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n👋 Window closed by user")
                    pygame.quit()
                    env.close()
                    return

            # Agent selects action
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Render is handled by the environment's render_mode="human"
            env.render()

        # Episode summary
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        print(f"✅ Episode {ep} finished:")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Steps: {steps}")
        print(f"   Paddle hits: {info.get('total_paddle_hits', 'N/A')}")
        print(f"   Bricks remaining: {info.get('bricks_remaining', 'N/A')}")
        print()

    # Final statistics
    print("=" * 50)
    print("📊 FINAL STATISTICS")
    print("=" * 50)
    print(f"Episodes played: {episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    print(f"Best episode: {np.max(episode_rewards):.2f}")
    print(f"Worst episode: {np.min(episode_rewards):.2f}")
    print("=" * 50)

    pygame.quit()
    env.close()
    print("\n✅ Visualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained PPO agent playing Breakout"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints_cnn_stable/ppo_breakout_cnn_1200000_steps",
        help="Path to trained model (without .zip extension)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic"
    )
    
    args = parser.parse_args()
    
    visualize(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()
