"""Tiny smoke test for the AE server/manager.

Run from this folder:
    python test_smoke.py
"""

from src.ae_manager import AEManager


def empty_cell():
    c = [0.0] * 25
    c[0] = 1.0  # visible
    c[5] = 1.0  # empty
    return c


def main():
    manager = AEManager()
    agent_view = [[empty_cell() for _ in range(5)] for _ in range(7)]
    base_view = [[empty_cell() for _ in range(5)] for _ in range(5)]

    obs = {
        "agent_viewcone": agent_view,
        "base_viewcone": base_view,
        "direction": 0,
        "location": [7, 7],
        "base_location": [7, 7],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 3,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 1],
    }

    action = manager.ae(obs)
    assert isinstance(action, int), action
    assert 0 <= action <= 5, action
    print({"action": action})


if __name__ == "__main__":
    main()
