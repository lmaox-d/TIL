"""Dump the local official novice map from til_environment.

Run after installing the official environment:

    cd ~/TIL/ae
    python dump_novice_map.py

Legend:
    A0..A5  agent starts
    B0..B5  team bases
    M       mission tile
    R       recon tile
    $       resource tile
    .       empty / no known static entity

This script uses the local til_environment state. It is useful because the
Novice environment is fixed in the official code path, but if the evaluator
secretly uses a different private fixed map, this dump is only a local proxy.
"""

from __future__ import annotations

from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from til_environment.entities import Mission, Recon, Resource


def main() -> None:
    cfg = default_config()
    cfg.env.num_teams = 6
    cfg.env.num_iters = 200
    cfg.env.novice = True
    cfg.env.render_mode = None

    env = Bomberman(cfg)
    env.reset(seed=88)
    dyn = env.dynamics

    grid = [["." for _ in range(dyn.grid_size)] for _ in range(dyn.grid_size)]

    for ent in dyn.registry.missions():
        x, y = map(int, ent.position)
        grid[y][x] = "M"
    for ent in dyn.registry.recons():
        x, y = map(int, ent.position)
        grid[y][x] = "R"
    for ent in dyn.registry.resources():
        x, y = map(int, ent.position)
        grid[y][x] = "$"

    print("Agent starts and bases:")
    for team in range(cfg.env.num_teams):
        agent = dyn.registry.get(f"agent_{team}")
        base = dyn.registry.get(f"base_team{team}")
        ax, ay = map(int, agent.position)
        bx, by = map(int, base.position)
        ad = int(agent.direction)
        print(f"team {team}: agent=({ax},{ay}) dir={ad} base=({bx},{by})")
        grid[ay][ax] = str(team)
        grid[by][bx] = "B"

    print("\nStatic tile map, y rows top-to-bottom, x columns left-to-right:")
    print("   " + " ".join(f"{x:2d}" for x in range(dyn.grid_size)))
    for y, row in enumerate(grid):
        print(f"{y:2d} " + "  ".join(row))

    print("\nKnown wall grid values, indexed as wall_grid[x][y] in decimal:")
    state = dyn.state
    for y in range(dyn.grid_size):
        print(" ".join(f"{int(state[x, y]):3d}" for x in range(dyn.grid_size)))

    env.close()


if __name__ == "__main__":
    main()
