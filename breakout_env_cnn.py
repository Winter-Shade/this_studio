import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import cv2

# =================== GAME CONSTANTS ===================

DISPLAYWIDTH = 640
DISPLAYHEIGHT = 480

BLOCKWIDTH = 56
BLOCKHEIGHT = 20
BLOCKGAP = 4
ARRAYWIDTH = 10
ARRAYHEIGHT = 5

PADDLEWIDTH = 100
PADDLEHEIGHT = 12
BALLRADIUS = 8

BALLSPEED = 6
PADDLE_SPEED = 10
MAX_BALL_SPEED = 12
MIN_BALL_SPEED = 4

# CNN observation constants (STANDARD ATARI SIZE)
OBS_WIDTH = 84
OBS_HEIGHT = 84


class BreakoutEnv(gym.Env):
    """
    CNN-Compatible Breakout Environment (GRAYSCALE - OPTIMIZED)

    Returns 84x84x1 grayscale images following DeepMind's Atari preprocessing.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ✅ GRAYSCALE observation space: 84x84x1 (OPTIMIZED)
        # Shape is (H, W, C) where C=1 for grayscale
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(OBS_HEIGHT, OBS_WIDTH, 1),  # ✅ 1 channel (grayscale)
            dtype=np.uint8
        )

        # Actions: 0=stay, 1=left, 2=right
        self.action_space = spaces.Discrete(3)

        # Game state tracking
        self.consecutive_hits = 0
        self.prev_ball_paddle_dist = None
        self.steps = 0
        self.total_paddle_hits = 0

        # Initialize pygame for rendering
        import pygame
        pygame.init()
        self.game_surface = pygame.Surface((DISPLAYWIDTH, DISPLAYHEIGHT))
        self.font = pygame.font.Font(None, 24)

        # For human rendering
        self.screen = None
        self.clock = None

    def _reset_ball_and_paddle(self):
        """Initialize with easier starting conditions"""
        half_pad = PADDLEWIDTH // 2

        # Center paddle
        self.paddle_x = float(DISPLAYWIDTH / 2)

        # 80% easy starts (ball near paddle, moving up)
        if random.random() < 0.8:
            self.ball_x = float(self.paddle_x + np.random.uniform(-50, 50))
            self.ball_y = float(DISPLAYHEIGHT - PADDLEHEIGHT - 60)

            speed = BALLSPEED * np.random.uniform(0.5, 0.7)
            self.vx = float(speed * np.random.uniform(-0.4, 0.4))
            self.vy = float(-speed)
        else:
            # 20% harder starts (ball coming down)
            self.ball_x = float(np.random.uniform(DISPLAYWIDTH * 0.3, DISPLAYWIDTH * 0.7))
            self.ball_y = float(np.random.uniform(100, DISPLAYHEIGHT * 0.4))

            speed = BALLSPEED * 0.8
            self.vx = float(np.random.uniform(-speed * 0.5, speed * 0.5))
            self.vy = float(speed * 0.6)

        self.consecutive_hits = 0
        self.prev_ball_paddle_dist = None

    def _reset_bricks(self):
        self.bricks = np.ones((ARRAYHEIGHT, ARRAYWIDTH), dtype=np.float32)

    def _render_game_surface(self):
        """Render the current game state to a pygame surface"""
        import pygame

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        BLUE = (0, 0, 255)
        GREEN = (0, 255, 0)

        # Clear surface
        self.game_surface.fill(BLACK)

        # Draw bricks
        for r in range(ARRAYHEIGHT):
            for c in range(ARRAYWIDTH):
                if self.bricks[r, c] == 1.0:
                    x = c * (BLOCKWIDTH + BLOCKGAP)
                    y = r * (BLOCKHEIGHT + BLOCKGAP)
                    pygame.draw.rect(self.game_surface, GREEN, (x, y, BLOCKWIDTH, BLOCKHEIGHT))

        # Draw paddle
        paddle_y = DISPLAYHEIGHT - PADDLEHEIGHT
        pygame.draw.rect(
            self.game_surface,
            BLUE,
            (self.paddle_x - PADDLEWIDTH // 2, paddle_y, PADDLEWIDTH, PADDLEHEIGHT)
        )

        # Draw ball (white for better contrast in grayscale)
        pygame.draw.circle(
            self.game_surface,
            WHITE,
            (int(self.ball_x), int(self.ball_y)),
            BALLRADIUS
        )

    def _get_obs(self):
        """
        Generate 84x84x1 GRAYSCALE observation
        
        This follows DeepMind's Atari preprocessing:
        1. Render game at native resolution
        2. Convert to grayscale
        3. Resize to 84x84
        4. Return as uint8 (0-255)
        """
        # Render game state to surface
        self._render_game_surface()

        # Convert pygame surface to numpy array (RGB)
        import pygame
        rgb_array = pygame.surfarray.array3d(self.game_surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))  # (H, W, C)

        # ✅ Convert RGB to GRAYSCALE
        # Using luminosity method: 0.299*R + 0.587*G + 0.114*B
        grayscale = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84 (standard Atari size)
        resized = cv2.resize(
            grayscale, 
            (OBS_WIDTH, OBS_HEIGHT), 
            interpolation=cv2.INTER_AREA
        )

        # Add channel dimension: (H, W) -> (H, W, 1)
        obs = resized[:, :, np.newaxis]

        return obs.astype(np.uint8)

    def _get_info(self):
        return {
            "consecutive_hits": self.consecutive_hits,
            "bricks_remaining": int(self.bricks.sum()),
            "total_paddle_hits": self.total_paddle_hits
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_ball_and_paddle()
        self._reset_bricks()
        self.steps = 0
        self.total_paddle_hits = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1

        # ---- PADDLE CONTROL ----
        if action == 1:  # Left
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:  # Right
            self.paddle_x += PADDLE_SPEED

        half_pad = PADDLEWIDTH // 2
        self.paddle_x = np.clip(self.paddle_x, half_pad, DISPLAYWIDTH - half_pad)

        # ---- BALL MOVEMENT ----
        self.ball_x += self.vx
        self.ball_y += self.vy

        reward = 0.0
        terminated = False
        truncated = False

        # ---- DISTANCE-BASED REWARD SHAPING ----
        ball_paddle_dist = abs(self.ball_x - self.paddle_x)

        # Reward for being aligned when ball is approaching
        if self.vy > 0 and self.ball_y > DISPLAYHEIGHT * 0.5:
            if ball_paddle_dist < 30:
                reward += 0.1
            elif ball_paddle_dist < 60:
                reward += 0.05

            # Reward for moving toward ball
            if self.prev_ball_paddle_dist is not None:
                if ball_paddle_dist < self.prev_ball_paddle_dist:
                    reward += 0.02
                else:
                    reward -= 0.01

        self.prev_ball_paddle_dist = ball_paddle_dist

        # ---- WALL BOUNCES ----
        if self.ball_x - BALLRADIUS <= 0:
            self.ball_x = BALLRADIUS
            self.vx = abs(self.vx)
        elif self.ball_x + BALLRADIUS >= DISPLAYWIDTH:
            self.ball_x = DISPLAYWIDTH - BALLRADIUS
            self.vx = -abs(self.vx)

        if self.ball_y - BALLRADIUS <= 0:
            self.ball_y = BALLRADIUS
            self.vy = abs(self.vy)

        # ---- BALL MISSED ----
        if self.ball_y - BALLRADIUS > DISPLAYHEIGHT:
            reward -= 10.0
            terminated = True

        # ---- PADDLE COLLISION ----
        paddle_y = DISPLAYHEIGHT - PADDLEHEIGHT

        if (self.vy > 0 and
            self.ball_y + BALLRADIUS >= paddle_y and
            self.ball_y - BALLRADIUS <= paddle_y + PADDLEHEIGHT and
            abs(self.ball_x - self.paddle_x) <= half_pad + BALLRADIUS):

            # Fix position
            self.ball_y = paddle_y - BALLRADIUS
            self.vy = -abs(self.vy)

            # Bounce angle
            offset = (self.ball_x - self.paddle_x) / half_pad
            offset = np.clip(offset, -0.85, 0.85)

            # Maintain speed
            current_speed = np.sqrt(self.vx**2 + self.vy**2)
            if current_speed < MIN_BALL_SPEED:
                current_speed = MIN_BALL_SPEED

            self.vx = current_speed * offset * 0.7
            self.vy = -np.sqrt(current_speed**2 - self.vx**2)

            # ---- PADDLE HIT REWARD ----
            self.consecutive_hits += 1
            self.total_paddle_hits += 1

            base_reward = 5.0
            streak_bonus = min(self.consecutive_hits - 1, 10) * 0.5
            reward += base_reward + streak_bonus

            # Curriculum: gradual speed increase
            if self.consecutive_hits >= 5 and self.consecutive_hits % 5 == 0:
                speed_mult = 1.02
                self.vx = np.clip(self.vx * speed_mult, -MAX_BALL_SPEED, MAX_BALL_SPEED)
                self.vy = np.clip(self.vy * speed_mult, -MAX_BALL_SPEED, MAX_BALL_SPEED)

        # ---- BRICK COLLISION ----
        brick_broken = False
        for r in range(ARRAYHEIGHT):
            for c in range(ARRAYWIDTH):
                if self.bricks[r, c] == 1.0:
                    brick_x = c * (BLOCKWIDTH + BLOCKGAP)
                    brick_y = r * (BLOCKHEIGHT + BLOCKGAP)

                    if (brick_x - BALLRADIUS <= self.ball_x <= brick_x + BLOCKWIDTH + BALLRADIUS and
                        brick_y - BALLRADIUS <= self.ball_y <= brick_y + BLOCKHEIGHT + BALLRADIUS):

                        self.bricks[r, c] = 0.0
                        brick_broken = True

                        # Bounce direction
                        dx = self.ball_x - (brick_x + BLOCKWIDTH / 2)
                        dy = self.ball_y - (brick_y + BLOCKHEIGHT / 2)

                        if abs(dx) > abs(dy):
                            self.vx *= -1
                        else:
                            self.vy *= -1

                        reward += 1.0
                        break

            if brick_broken:
                break

        # ---- WIN CONDITION ----
        if self.bricks.sum() == 0:
            reward += 100.0
            terminated = True

        # ---- TIME LIMIT ----
        if self.steps >= 5000:
            truncated = True

        # ---- SURVIVAL BONUS ----
        if not terminated:
            reward += 0.01

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # =================== RENDERING ===================

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_human(self):
        """Render for human viewing (full color)"""
        import pygame

        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.display.set_mode((DISPLAYWIDTH, DISPLAYHEIGHT))
            pygame.display.set_caption("Breakout CNN - Grayscale Training")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Render game state
        self._render_game_surface()

        # Blit to display
        self.screen.blit(self.game_surface, (0, 0))

        # Add info text
        WHITE = (255, 255, 255)
        info_text = self.font.render(
            f"Hits: {self.total_paddle_hits}  Bricks: {int(self.bricks.sum())}  Steps: {self.steps}",
            True, WHITE
        )
        self.screen.blit(info_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def _render_rgb_array(self):
        """Return RGB array of current game state"""
        self._render_game_surface()
        import pygame
        rgb_array = pygame.surfarray.array3d(self.game_surface)
        return np.transpose(rgb_array, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


# =================== WHY GRAYSCALE? ===================
#
# 1. MEMORY EFFICIENCY:
#    - RGB: 84 × 84 × 3 × 4 frames = 84,672 values per observation
#    - Grayscale: 84 × 84 × 1 × 4 frames = 28,224 values per observation
#    → 3x less memory!
#
# 2. TRAINING SPEED:
#    - Fewer input channels → fewer conv filter parameters
#    - RGB CNN: ~3M parameters
#    - Grayscale CNN: ~1M parameters
#    → 3x faster training!
#
# 3. NO INFORMATION LOSS:
#    - Breakout gameplay doesn't depend on color
#    - Ball position, paddle position, brick positions all visible in grayscale
#    - Motion from frame stacking works the same
#
# 4. STANDARD PRACTICE:
#    - DeepMind's DQN paper (Nature 2015) uses grayscale
#    - All Atari baselines use grayscale
#    - Industry standard for Atari RL
#
# 5. BETTER GENERALIZATION:
#    - Color can be distracting noise
#    - Grayscale forces network to focus on shapes/positions
#    - Often leads to better final performance
#
# When to use RGB instead?
# - Games where color is gameplay-critical (e.g., color matching games)
# - Real-world robotics with complex textures
# - NOT for simple games like Breakout!