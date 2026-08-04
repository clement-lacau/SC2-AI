import time
import concurrent.futures
from collections import defaultdict

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from bots.rusher import CombatBot
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.data import Race, Difficulty, Result
from sc2 import maps

def play_match(params):
    enemy_race, enemy_difficulty = params
    result = run_game(
        maps.get("AutomatonLE"),
        [Bot(Race.Protoss, CombatBot()), Computer(enemy_race, enemy_difficulty)],
        realtime=False
    )
    return (f"{enemy_race.name} ({enemy_difficulty.name})", result)

if __name__ == "__main__":
    opponents = [
        (Race.Terran, Difficulty.Easy),
        (Race.Zerg, Difficulty.Easy),
        (Race.Protoss, Difficulty.Easy),
        (Race.Terran, Difficulty.Medium),
        (Race.Zerg, Difficulty.Medium),
        (Race.Protoss, Difficulty.Medium),
    ]
    
    iterations_per_enemy = 2
    matches_to_play = opponents * iterations_per_enemy 

    results_data = defaultdict(lambda: {"Victoires": 0, "Défaites": 0, "Égalités": 0})
    total_matches = len(matches_to_play)
    
    print(f"Lancement de {total_matches} parties ")
    start_time = time.time()

    coeurs = max(1, (os.cpu_count() or 2) // 2)
    print(f"Utilisation de {coeurs} coeurs sur {os.cpu_count()} disponibles...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=coeurs) as executor:
        results = executor.map(play_match, matches_to_play)

    total_victories = 0
    for opponent_str, result in results:
        if result == Result.Victory:
            results_data[opponent_str]["Victoires"] += 1
            total_victories += 1
        elif result == Result.Defeat:
            results_data[opponent_str]["Défaites"] += 1
        else:
            results_data[opponent_str]["Égalités"] += 1

    win_rate = (total_victories / total_matches) * 100
    duration = round(time.time() - start_time, 1)

    print("\n=== RÉSULTATS DU BENCHMARK ===")
    print(f"**Temps d'exécution** : {duration} secondes")
    print(f"**Taux de victoire global** : {win_rate:.1f}% ({total_victories}/{total_matches})\n")
    
    print("| Adversaire | Victoires | Défaites | Taux (Winrate) |")
    print("|---|---|---|---|")
    
    for opp in sorted(results_data.keys()):
        stats = results_data[opp]
        v = stats["Victoires"]
        d = stats["Défaites"]
        total_opp = v + d + stats["Égalités"]
        wr_opp = (v / total_opp) * 100 if total_opp > 0 else 0
        
        print(f"| {opp} | {v} | {d} | {wr_opp:.1f}% |")