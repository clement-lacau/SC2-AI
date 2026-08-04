import sys
import time
import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from sc2.main import _host_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class ProgressLoggerCallback(BaseCallback):
    def __init__(self, check_freq=100):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            mean_reward = np.mean(self.training_env.get_attr("bot")[0].minerals) if hasattr(self.training_env.get_attr("bot")[0], "minerals") else 0
            print(f"| Timesteps: {self.num_timesteps}/{self.locals['total_timesteps']} | Temps: {int(elapsed)}s | Vitesse: {fps} steps/s |")
        return True

class RLBot(BotAI):
    def __init__(self):
        super().__init__()
        self.action_to_do = None

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        
        if self.action_to_do == 0:
            for nexus in self.townhalls.ready.idle:
                if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0:
                    nexus.train(UnitTypeId.PROBE)
        elif self.action_to_do == 1:
            if self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                if self.townhalls.ready.exists:
                    await self.build(UnitTypeId.PYLON, near=self.townhalls.ready.first)
        elif self.action_to_do == 2:
            pylons = self.structures(UnitTypeId.PYLON).ready
            if pylons.exists and self.can_afford(UnitTypeId.GATEWAY):
                await self.build(UnitTypeId.GATEWAY, near=pylons.random)
        elif self.action_to_do == 3:
            for gw in self.structures(UnitTypeId.GATEWAY).ready.idle:
                if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0:
                    gw.train(UnitTypeId.ZEALOT)
        elif self.action_to_do == 4:
            zealots = self.units(UnitTypeId.ZEALOT)
            if zealots.exists and self.enemy_start_locations:
                for z in zealots.idle:
                    z.attack(self.enemy_start_locations[0])

class SC2Env(gym.Env):
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=10000, shape=(5,), dtype=np.float32)
        self.bot = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bot = RLBot()
        
        self.game_task = self.loop.create_task(
            _host_game(
                maps.get("AutomatonLE"),
                [Bot(Race.Protoss, self.bot), Computer(Race.Terran, Difficulty.Easy)],
                realtime=False
            )
        )
        
        while not (hasattr(self.bot, "state") and self.bot.state is not None):
            self.loop.run_until_complete(asyncio.sleep(0.05))
            
        return self._get_obs(), {}

    def _get_obs(self):
        game_time = self.bot.time / 60 if hasattr(self.bot, "state") and self.bot.state is not None else 0.0
        return np.array([
            float(self.bot.minerals),
            float(self.bot.workers.amount),
            float(self.bot.structures(UnitTypeId.GATEWAY).amount),
            float(self.bot.units(UnitTypeId.ZEALOT).amount),
            float(game_time)
        ], dtype=np.float32)

    def step(self, action):
        self.bot.action_to_do = action
        
        self.loop.run_until_complete(asyncio.sleep(0.01))
        
        obs = self._get_obs()
        reward = (self.bot.workers.amount * 1) + (self.bot.units(UnitTypeId.ZEALOT).amount * 5)
        done = self.game_task.done()
        
        return obs, reward, done, False, {}

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    env = SC2Env(loop)
    model = PPO("MlpPolicy", env, verbose=0)
    
    logger_callback = ProgressLoggerCallback(check_freq=50)
    
    print("Début de l'entraînement ML...")
    model.learn(total_timesteps=5000, callback=logger_callback)
    model.save("sc2_rl_model")
    print("Modèle sauvegardé sous 'sc2_rl_model.zip'")