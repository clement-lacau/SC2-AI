import matplotlib.pyplot as plt
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Result
from sc2 import maps
from sc2.ids.unit_typeid import UnitTypeId

class CombatBot(BotAI):
    def __init__(self):
        super().__init__()
        self.history_time = []
        self.history_minerals = []
        self.history_army = []

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        
        nexus = self.townhalls.ready.random if self.townhalls.ready else None
        
        if nexus and nexus.is_idle and self.workers.amount < 22:
            if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0:
                nexus.train(UnitTypeId.PROBE)
        
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            if nexus and self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=nexus)
        
        pylons = self.structures(UnitTypeId.PYLON).ready
        if pylons.exists:
            pylon = pylons.random
            if self.structures(UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY) < 4:
                if self.can_afford(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)
        
        for gw in self.structures(UnitTypeId.GATEWAY).ready.idle:
            if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0:
                gw.train(UnitTypeId.ZEALOT)
        
        zealots = self.units(UnitTypeId.ZEALOT)
        if zealots.amount >= 15:
            for zealot in zealots.idle:
                zealot.attack(self.enemy_start_locations[0])

        if iteration % 112 == 0:
            self.history_time.append(self.time / 60)
            self.history_minerals.append(self.minerals)
            self.history_army.append(zealots.amount)

    async def on_end(self, game_result: Result):
        print("\n| Métrique | Valeur |")
        print("|---|---|")
        print(f"| **Résultat** | {game_result.name} |")
        print(f"| **Durée** | {self.time_formatted} |")
        print(f"| **Zélotes max** | {max(self.history_army) if self.history_army else 0} |")
        
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        ax1.plot(self.history_time, self.history_minerals, color='blue', label='Minerai')
        ax1.set_xlabel("Temps (minutes)")
        ax1.set_ylabel("Minerai", color='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history_time, self.history_army, color='red', label='Zélotes')
        ax2.set_ylabel("Taille de l'armée", color='red')
        
        plt.title(f"Ressources vs Armée ({game_result.name})")
        plt.grid(True, alpha=0.3)
        plt.savefig("graphe_combat.png")
        print("\n=> Graphique sauvegardé sous 'graphe_combat.png'")

if __name__ == "__main__":
    run_game(
        maps.get("AutomatonLE"),
        [Bot(Race.Protoss, CombatBot()), Computer(Race.Terran, Difficulty.Easy)], 
        realtime=False
    )