"""Evaluate the AE agent inside the official til_environment.

Run from TIL/ae after installing til_environment:
    python evaluate_rule_agent.py --episodes 10 --novice
    python evaluate_rule_agent.py --episodes 1 --novice --details
    python evaluate_rule_agent.py --episodes 10 --novice --no-bombs
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ae_manager import AEManager  # noqa: E402

PLACE_BOMB = 5
STAY = 4


def make_env(novice: bool = False, render: bool = False):
    from til_environment.bomberman_env import Bomberman
    from til_environment.config import default_config

    cfg = default_config()
    cfg.env.num_teams = 6
    cfg.env.num_iters = 200
    cfg.env.novice = bool(novice)
    cfg.env.render_mode = "rgb_array" if render else None
    cfg.renderer.debug = False
    return Bomberman(cfg)


def run_episode(seed: int, novice: bool = False, render: bool = False, no_bombs: bool = False) -> dict[str, Any]:
    env = make_env(novice=novice, render=render)
    env.reset(seed=seed)

    managers = {agent: AEManager() for agent in env.agents}
    totals = {agent: 0.0 for agent in env.agents}
    action_counts = {agent: {i: 0 for i in range(6)} for agent in env.agents}

    max_turns = 6 * 205
    turns = 0
    while env.agents and turns < max_turns:
        agent = env.agent_selection
        obs = env.observe(agent)
        term = bool(env.terminations.get(agent, False))
        trunc = bool(env.truncations.get(agent, False))

        if term or trunc:
            action = None
        else:
            action = managers[agent].ae(obs)
            if no_bombs and action == PLACE_BOMB:
                action = STAY
            action_counts[agent][int(action)] += 1

        env.step(action)

        if getattr(env.agent_selector, "is_first", lambda: False)():
            for a, r in env.rewards.items():
                totals[a] = totals.get(a, 0.0) + float(r)

        turns += 1
        if all(env.truncations.values()) or all(env.terminations.values()):
            break

    env.close()
    return {
        "seed": seed,
        "total_by_agent": totals,
        "action_counts": action_counts,
        "mean_reward": sum(totals.values()) / max(1, len(totals)),
        "max_reward": max(totals.values()) if totals else 0.0,
        "min_reward": min(totals.values()) if totals else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--novice", action="store_true", help="use fixed novice map")
    parser.add_argument("--render", action="store_true", help="slow; render rgb frames")
    parser.add_argument("--details", action="store_true", help="print per-agent totals and action counts")
    parser.add_argument("--no-bombs", action="store_true", help="replace PLACE_BOMB with STAY for ablation testing")
    args = parser.parse_args()

    results = []
    for i in range(args.episodes):
        result = run_episode(seed=args.seed + i, novice=args.novice, render=args.render, no_bombs=args.no_bombs)
        results.append(result)
        print(
            f"episode={i:03d} seed={result['seed']} "
            f"mean={result['mean_reward']:.2f} max={result['max_reward']:.2f} min={result['min_reward']:.2f}"
        )
        if args.details:
            for agent, total in sorted(result["total_by_agent"].items()):
                counts = result["action_counts"][agent]
                print(f"  {agent}: reward={total:.2f} actions={counts}")

    means = [r["mean_reward"] for r in results]
    print("\nSummary")
    print(f"episodes: {len(results)}")
    print(f"mean reward: {statistics.mean(means):.3f}")
    print(f"median reward: {statistics.median(means):.3f}")
    if len(means) > 1:
        print(f"stdev: {statistics.stdev(means):.3f}")


if __name__ == "__main__":
    main()
