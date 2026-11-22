"""Atari-100k style environment on top of ALE: action repeat, max-pool of last two frames, resize.

No sticky actions and no reward clipping. `is_terminal` is a real game end;
`is_last` also covers truncation at `max_frames`.
"""
import ale_py
import ale_py.roms
import numpy as np
from PIL import Image


class Atari:
    def __init__(self, game="pong", size=64, repeat=4, noops=30, max_frames=108000, seed=0):
        self.size, self.repeat, self.noops, self.max_frames = size, repeat, noops, max_frames
        self.rng = np.random.default_rng(seed)
        self.ale = ale_py.ALEInterface()
        self.ale.setLoggerMode(ale_py.LoggerMode.Error)
        self.ale.setInt("random_seed", int(self.rng.integers(0, 2**31 - 1)))
        self.ale.setFloat("repeat_action_probability", 0.0)  # no sticky actions
        self.ale.setInt("frame_skip", 1)                      # frame skip is done in step()
        self.ale.loadROM(ale_py.roms.get_rom_path(game))
        self.actions = self.ale.getMinimalActionSet()          # Pong: 6 actions
        self.num_actions = len(self.actions)
        self.frames = 0
        self.done = True

    def _screen(self):
        return self.ale.getScreenRGB()

    def _obs(self, image, reward, is_first=False, is_last=False, is_terminal=False):
        image = Image.fromarray(image).resize((self.size, self.size), Image.BILINEAR)
        return dict(image=np.asarray(image, dtype=np.uint8), reward=np.float32(reward),
                    is_first=is_first, is_last=is_last, is_terminal=is_terminal)

    def reset(self):
        self.ale.reset_game()
        for _ in range(int(self.rng.integers(self.noops + 1))):  # random no-ops diversify the initial state
            self.ale.act(self.actions[0])
        self.frames, self.done = 0, False
        return self._obs(self._screen(), 0.0, is_first=True)

    def step(self, action):
        assert not self.done, "Call reset() after the episode ends."
        reward, last_two = 0.0, []
        for _ in range(self.repeat):
            reward += self.ale.act(self.actions[int(action)])
            self.frames += 1
            last_two = (last_two + [self._screen()])[-2:]
            if self.ale.game_over():
                break
        terminal = self.ale.game_over()
        truncated = self.frames >= self.max_frames
        self.done = terminal or truncated
        image = np.max(last_two, axis=0)  # max over last two frames removes Atari sprite flicker
        return self._obs(image, reward, is_last=self.done, is_terminal=terminal)
