import matplotlib.pyplot as plt
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Result
from sc2 import maps
from sc2.ids.unit_typeid import UnitTypeId

class StatsBot(BotAI):
    def __init__(self):
        super().__init__()
        self.history_time = []
        self.history_minerals = []
        self.history_probes = []

    async def on_step(self, iteration: int):
        await self.distribute_workers()
        
        for nexus in self.townhalls.ready.idle:
            if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0:
                nexus.train(UnitTypeId.PROBE)

        # Enregistrement des données toutes les ~5 secondes (112 frames)
        if iteration % 112 == 0:
            self.history_time.append(self.time / 60)
            self.history_minerals.append(self.minerals)
            self.history_probes.append(self.workers.amount)

    async def on_end(self, game_result: Result):
        print("\n| Métrique | Valeur |")
        print("|---|---|")
        print(f"| **Durée** | {self.time_formatted} |")
        print(f"| **Sondes** | {self.workers.amount} |")
        print(f"| **Minerai**| {self.minerals} |")
        
        plt.figure(figsize=(10, 5))
        
        ax1 = plt.gca()
        ax1.plot(self.history_time, self.history_minerals, color='blue', label='Minerai')
        ax1.set_xlabel("Temps (minutes)")
        ax1.set_ylabel("Minerai", color='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history_time, self.history_probes, color='orange', label='Sondes')
        ax2.set_ylabel("Sondes actives", color='orange')
        
        plt.title("Évolution des ressources et unités")
        plt.grid(True, alpha=0.3)
        plt.savefig("graphe.png")
        print("\n=> Graphique sauvegardé sous 'graphe.png'")

if __name__ == "__main__":
    run_game(
        maps.get("AutomatonLE"),
        [Bot(Race.Protoss, StatsBot()), Computer(Race.Terran, Difficulty.Easy)], 
        realtime=False
    )