import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

sys.path.append(str(Path(__file__).resolve().parent))

from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

class TrainedBot(BotAI):
    def __init__(self, model_path):
        super().__init__()
        self.model = PPO.load(model_path)

    async def on_step(self, iteration: int):
        await self.distribute_workers()

        obs = np.array([
            float(self.minerals),
            float(self.workers.amount),
            float(self.structures(UnitTypeId.GATEWAY).amount),
            float(self.units(UnitTypeId.ZEALOT).amount),
            float(self.time / 60)
        ], dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)

        if action == 0:
            for nexus in self.townhalls.ready.idle:
                if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0:
                    nexus.train(UnitTypeId.PROBE)
        elif action == 1:
            if self.can_afford(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                if self.townhalls.ready.exists:
                    await self.build(UnitTypeId.PYLON, near=self.townhalls.ready.first)
        elif action == 2:
            pylons = self.structures(UnitTypeId.PYLON).ready
            if pylons.exists and self.can_afford(UnitTypeId.GATEWAY):
                await self.build(UnitTypeId.GATEWAY, near=pylons.random)
        elif action == 3:
            for gw in self.structures(UnitTypeId.GATEWAY).ready.idle:
                if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0:
                    gw.train(UnitTypeId.ZEALOT)
        elif action == 4:
            zealots = self.units(UnitTypeId.ZEALOT)
            if zealots.exists and self.enemy_start_locations:
                for z in zealots.idle:
                    z.attack(self.enemy_start_locations[0])

if __name__ == "__main__":
    run_game(
        maps.get("AutomatonLE"),
        [Bot(Race.Protoss, TrainedBot("sc2_rl_model")), Computer(Race.Terran, Difficulty.Easy)],
        realtime=False
    )