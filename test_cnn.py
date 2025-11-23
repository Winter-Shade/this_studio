"""
Test CNN model - Headless compatible
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Set headless mode for servers
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from breakout_env_cnn import BreakoutEnv


def make_env(render=False):
    """Create environment matching training setup"""
    # Note: render=True still works in headless mode (just no visual)
    env = BreakoutEnv(render_mode=None)  # Always None for testing
    env = Monitor(env)
    return env


def test_agent(model_path, num_episodes=20):
    """Test CNN agent with proper normalization"""
    
    print("=" * 70)
    print(f"TESTING: {model_path}")
    print("=" * 70)
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded")
        print(f"  Device: {model.device}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return None
    
    # Create environment
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Try to load normalization stats
    norm_paths = [
        "vec_normalize_cnn_stable.pkl",
        "vec_normalize_cnn_stable_interrupted.pkl",
        f"{os.path.dirname(model_path)}/vec_normalize_cnn_stable.pkl",
    ]
    
    norm_loaded = False
    for norm_path in norm_paths:
        if os.path.exists(norm_path):
            try:
                env = VecNormalize.load(norm_path, env)
                env.training = False
                env.norm_reward = False
                print(f"✓ Normalization loaded: {norm_path}")
                norm_loaded = True
                break
            except Exception as e:
                print(f"  Warning: Could not load {norm_path}: {e}")
    
    if not norm_loaded:
        print("⚠ No normalization file found - testing without normalization")
        print("  (Results may be inaccurate)")
    
    # Test
    stats = {
        "rewards": [],
        "lengths": [],
        "hits": [],
        "bricks": [],
        "wins": 0
    }
    
    print(f"\nTesting {num_episodes} episodes...")
    print("-" * 70)
    
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            ep_reward += reward[0]
            steps += 1
            
            if done[0]:
                break
        
        # Record stats
        hits = info[0].get("total_paddle_hits", 0)
        bricks_left = info[0].get("bricks_remaining", 50)
        bricks_broken = 50 - bricks_left
        won = bricks_left == 0
        
        stats["rewards"].append(ep_reward)
        stats["lengths"].append(steps)
        stats["hits"].append(hits)
        stats["bricks"].append(bricks_broken)
        if won:
            stats["wins"] += 1
        
        status = "WIN ✓" if won else "LOSS"
        print(f"Ep {ep+1:2d}: {status:6s} | "
              f"R: {ep_reward:6.1f} | "
              f"Steps: {steps:4d} | "
              f"Hits: {hits:3d} | "
              f"Bricks: {bricks_broken:2d}/50")
    
    # Summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Win Rate:          {stats['wins']}/{num_episodes} ({100*stats['wins']/num_episodes:.1f}%)")
    print(f"Avg Reward:        {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    print(f"Avg Paddle Hits:   {np.mean(stats['hits']):.1f}")
    print(f"Avg Bricks Broken: {np.mean(stats['bricks']):.1f}/50")
    print(f"Max Paddle Hits:   {max(stats['hits'])}")
    print(f"Max Bricks Broken: {max(stats['bricks'])}/50")
    print("=" * 70)
    
    # Performance analysis
    avg_hits = np.mean(stats['hits'])
    win_rate = stats['wins'] / num_episodes
    
    print("\nPERFORMANCE ANALYSIS:")
    if avg_hits < 5:
        print("  ⚠ Poor - Agent struggles with paddle control")
        print("    → Continue training or check reward structure")
    elif avg_hits < 15:
        print("  ⚪ Fair - Agent learning paddle control")
        print("    → Training in progress, showing improvement")
    elif avg_hits < 30:
        print("  ✓ Good - Agent has solid paddle control")
        print("    → CNN learning trajectory patterns")
    else:
        print("  ✓✓ Excellent - Agent mastering the game!")
        print("    → Frame stacking working well")
    
    if win_rate == 0:
        print("  ⚠ No wins yet - needs more training")
    elif win_rate < 0.2:
        print("  ⚪ Some wins - making progress")
    elif win_rate < 0.5:
        print("  ✓ Good win rate - performing well")
    else:
        print("  ✓✓ High win rate - excellent performance!")
    
    print("=" * 70)
    
    env.close()
    return stats


def compare_checkpoints():
    """Compare multiple checkpoints to find best"""
    import glob
    
    checkpoints = glob.glob("checkpoints_cnn_stable/*.zip")
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print("\n" + "=" * 70)
    print("COMPARING CHECKPOINTS")
    print("=" * 70)
    
    results = {}
    for ckpt in sorted(checkpoints):
        print(f"\nTesting: {ckpt}")
        stats = test_agent(ckpt, num_episodes=5)  # Quick test
        if stats:
            results[ckpt] = {
                'avg_reward': np.mean(stats['rewards']),
                'avg_hits': np.mean(stats['hits']),
                'win_rate': stats['wins'] / len(stats['rewards'])
            }
    
    # Find best
    if results:
        print("\n" + "=" * 70)
        print("CHECKPOINT COMPARISON")
        print("=" * 70)
        best_ckpt = max(results.items(), key=lambda x: x[1]['avg_reward'])
        print(f"\nBest checkpoint: {best_ckpt[0]}")
        print(f"  Avg Reward: {best_ckpt[1]['avg_reward']:.2f}")
        print(f"  Avg Hits: {best_ckpt[1]['avg_hits']:.1f}")
        print(f"  Win Rate: {best_ckpt[1]['win_rate']*100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_breakout_cnn_stable_final")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--compare", action="store_true", 
                       help="Compare all checkpoints")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_checkpoints()
    else:
        test_agent(args.model, args.episodes)


# USAGE:
# 
# Test final model:
#   python test_cnn_stable.py --model ppo_breakout_cnn_stable_final --episodes 50
#
# Test specific checkpoint:
#   python test_cnn_stable.py --model checkpoints_cnn_stable/ppo_breakout_cnn_500000_steps
#
# Compare all checkpoints:
#   python test_cnn_stable.py --compare
#
# Test interrupted model:
#   python test_cnn_stable.py --model ppo_breakout_cnn_stable_interrupted