"""Fast novice-focused AE agent for TIL-AI 2026.

This is a lightweight deterministic planner, not a neural network. It keeps an
internal map from partial observations, chases high-value tiles, bombs useful
walls/enemy bases, and avoids known bomb blasts.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Any

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
OPP = (2, 3, 0, 1)

VISIBLE = 0
WALL_RIGHT = 1
WALL_DOWN = 2
WALL_LEFT = 3
WALL_UP = 4
TILE_EMPTY = 5
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ENEMY_AGENT = 10
ENEMY_BASE = 12
DESTR_RIGHT = 13
DESTR_DOWN = 14
DESTR_LEFT = 15
DESTR_UP = 16
ALLY_BOMB = 17
ENEMY_BOMB = 18
ALLY_BOMB_TIMER = 19
ENEMY_BOMB_TIMER = 20
ENEMY_BASE_HEALTH = 24

GRID = 16
VIEW_LEFT = 2
VIEW_BEHIND = 2
BOMB_TIMER = 4
BOMB_RADIUS = 2


def _pair(value: Any, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        return int(value[0]), int(value[1])
    except Exception:
        return default


def _ch(cell: Any, idx: int) -> float:
    try:
        return float(cell[idx])
    except Exception:
        return 0.0


def _legal(mask: list[int], action: int) -> bool:
    return 0 <= action < len(mask) and bool(mask[action])


class AEManager:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.walls: dict[tuple[int, int], list[bool | None]] = defaultdict(lambda: [None, None, None, None])
        self.destructible: dict[tuple[int, int], list[bool | None]] = defaultdict(lambda: [None, None, None, None])
        self.items: dict[tuple[int, int], tuple[str, int]] = {}
        self.enemy_bases: dict[tuple[int, int], float] = {}
        self.enemy_agents: dict[tuple[int, int], int] = {}
        self.bombs: dict[tuple[int, int], tuple[int, int]] = {}
        self.seen: set[tuple[int, int]] = set()
        self.visited: dict[tuple[int, int], int] = defaultdict(int)
        self.last_step = -1
        self.last_bomb_step = -99
        self.evade_until = -1

    def ae(self, observation: dict[str, Any]) -> int:
        step = int(observation.get("step", 0))
        if step == 0 or (self.last_step >= 0 and step < self.last_step):
            self.reset()

        loc = _pair(observation.get("location", [0, 0]))
        direction = int(observation.get("direction", 0)) % 4
        mask = [int(x) for x in observation.get("action_mask", [1, 1, 1, 1, 1, 0])]

        self._age(step)
        self._update_from_obs(observation, step)
        self.visited[loc] += 1
        self.last_step = step

        if not any(mask):
            return STAY
        if int(observation.get("frozen_ticks", 0)) > 0:
            return STAY if _legal(mask, STAY) else self._first_legal(mask)

        danger_now, danger_soon = self._danger(step)
        if loc in danger_now or (self.evade_until >= step and loc in danger_soon):
            safe = self._nearest_safe(loc, danger_soon)
            return self._fallback(self._action_to(loc, direction, safe, mask, danger_soon), loc, direction, mask, danger_soon)

        if _legal(mask, PLACE_BOMB) and self._should_bomb(loc, step, danger_soon):
            self.bombs[loc] = (BOMB_TIMER, step)
            self.last_bomb_step = step
            self.evade_until = step + BOMB_TIMER + 1
            return PLACE_BOMB

        target = self._target(loc, step, danger_soon)
        return self._fallback(self._action_to(loc, direction, target, mask, danger_soon), loc, direction, mask, danger_soon)

    def _update_from_obs(self, obs: dict[str, Any], step: int) -> None:
        loc = _pair(obs.get("location", [0, 0]))
        direction = int(obs.get("direction", 0)) % 4
        self._read_view(obs.get("agent_viewcone"), loc, direction, step, True)
        self._read_view(obs.get("base_viewcone"), _pair(obs.get("base_location", [0, 0])), 0, step, False)

    def _read_view(self, view: Any, origin: tuple[int, int], direction: int, step: int, agent_view: bool) -> None:
        # Official env gives numpy arrays. Never use `view or []`.
        if view is None:
            return
        try:
            side = len(view)
        except Exception:
            return
        for i, row in enumerate(view):
            for j, cell in enumerate(row):
                if _ch(cell, VISIBLE) < 0.5:
                    continue
                if agent_view:
                    pos = self._view_to_world(origin, direction, (i - VIEW_BEHIND, j - VIEW_LEFT))
                else:
                    r = side // 2
                    pos = (origin[0] + i - r, origin[1] + j - r)
                if self._in_bounds(pos):
                    self._record(pos, cell, step)

    @staticmethod
    def _view_to_world(origin: tuple[int, int], direction: int, rel: tuple[int, int]) -> tuple[int, int]:
        x, y = origin
        a, b = rel
        if direction == 0:
            return x + a, y + b
        if direction == 1:
            return x - b, y + a
        if direction == 2:
            return x - a, y - b
        return x + b, y - a

    def _record(self, pos: tuple[int, int], cell: Any, step: int) -> None:
        self.seen.add(pos)
        for d, wc in enumerate((WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP)):
            blocked = _ch(cell, wc) >= 0.5
            destr = _ch(cell, (DESTR_RIGHT, DESTR_DOWN, DESTR_LEFT, DESTR_UP)[d]) >= 0.5
            self._set_wall(pos, d, blocked, destr)

        if _ch(cell, TILE_MISSION) >= 0.5:
            self.items[pos] = ("mission", step)
        elif _ch(cell, TILE_RESOURCE) >= 0.5:
            self.items[pos] = ("resource", step)
        elif _ch(cell, TILE_RECON) >= 0.5:
            self.items[pos] = ("recon", step)
        elif _ch(cell, TILE_EMPTY) >= 0.5:
            self.items.pop(pos, None)

        if _ch(cell, ENEMY_BASE) >= 0.5:
            hp = _ch(cell, ENEMY_BASE_HEALTH)
            self.enemy_bases[pos] = hp if hp > 0 else 1.0
        if _ch(cell, ENEMY_AGENT) >= 0.5:
            self.enemy_agents[pos] = step
        if _ch(cell, ALLY_BOMB) >= 0.5:
            self.bombs[pos] = (int(_ch(cell, ALLY_BOMB_TIMER) or BOMB_TIMER), step)
        if _ch(cell, ENEMY_BOMB) >= 0.5:
            self.bombs[pos] = (int(_ch(cell, ENEMY_BOMB_TIMER) or BOMB_TIMER), step)

    def _set_wall(self, pos: tuple[int, int], d: int, blocked: bool, destr: bool) -> None:
        self.walls[pos][d] = blocked
        self.destructible[pos][d] = destr if blocked else False
        nb = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
        if self._in_bounds(nb):
            self.walls[nb][OPP[d]] = blocked
            self.destructible[nb][OPP[d]] = destr if blocked else False

    def _age(self, step: int) -> None:
        for pos, (timer, seen) in list(self.bombs.items()):
            if timer - max(0, step - seen) <= 0:
                self.bombs.pop(pos, None)
        for pos, seen in list(self.enemy_agents.items()):
            if step - seen > 6:
                self.enemy_agents.pop(pos, None)

    def _danger(self, step: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        now, soon = set(), set()
        for pos, (timer, seen) in self.bombs.items():
            est = timer - max(0, step - seen)
            cells = self._blast(pos)
            if est <= 2:
                now.update(cells)
            if est <= BOMB_TIMER:
                soon.update(cells)
        return now, soon

    def _blast(self, pos: tuple[int, int]) -> set[tuple[int, int]]:
        out = {pos}
        bx, by = pos
        for x in range(max(0, bx - BOMB_RADIUS), min(GRID, bx + BOMB_RADIUS + 1)):
            for y in range(max(0, by - BOMB_RADIUS), min(GRID, by + BOMB_RADIUS + 1)):
                p = (x, y)
                if self._line_clear(pos, p, unknown_blocks=False):
                    out.add(p)
        return out

    def _should_bomb(self, loc: tuple[int, int], step: int, danger: set[tuple[int, int]]) -> bool:
        if step - self.last_bomb_step < 5:
            return False
        blast = self._blast(loc)
        if self._nearest_safe(loc, danger | blast, 7) is None:
            return False
        if any(base in blast for base in self.enemy_bases):
            return True
        if any(step - seen <= 2 and enemy in blast for enemy, seen in self.enemy_agents.items()):
            return True
        return any(self.walls[loc][d] is True and self.destructible[loc][d] is True for d in range(4)) and self.visited[loc] >= 2

    def _target(self, loc: tuple[int, int], step: int, danger: set[tuple[int, int]]) -> tuple[int, int] | None:
        best: tuple[float, tuple[int, int]] | None = None
        value = {"mission": 5.0, "resource": 2.0, "recon": 1.0}
        for pos, (kind, seen) in list(self.items.items()):
            dist = self._dist(loc, pos)
            if dist is None:
                continue
            score = value.get(kind, 0.5) * 100 - 4.0 * dist - 0.03 * (step - seen) - 0.8 * self.visited[pos]
            score += 8.0 / (1.0 + self._centre_dist(pos))
            if pos in danger:
                score -= 1000
            if best is None or score > best[0]:
                best = (score, pos)

        for base in self.enemy_bases:
            for pos in self._bomb_cells_for(base):
                if pos in danger:
                    continue
                dist = self._dist(loc, pos)
                if dist is None:
                    continue
                score = 230 - 5.0 * dist
                if best is None or score > best[0]:
                    best = (score, pos)

        if best is not None:
            return best[1]

        best_frontier: tuple[float, tuple[int, int]] | None = None
        for pos in self.seen:
            if pos in danger:
                continue
            unseen = sum(1 for d, nb in self._dir_neighbours(pos) if nb not in self.seen and self._can_step(pos, d, True))
            if unseen == 0:
                continue
            dist = self._dist(loc, pos)
            if dist is None:
                continue
            score = 20 * unseen - 2.5 * dist - self.visited[pos] + 5 / (1 + self._centre_dist(pos))
            if best_frontier is None or score > best_frontier[0]:
                best_frontier = (score, pos)
        if best_frontier is not None:
            return best_frontier[1]
        return (GRID // 2, GRID // 2)

    def _bomb_cells_for(self, base: tuple[int, int]) -> list[tuple[int, int]]:
        bx, by = base
        out = []
        for x in range(max(0, bx - BOMB_RADIUS), min(GRID, bx + BOMB_RADIUS + 1)):
            for y in range(max(0, by - BOMB_RADIUS), min(GRID, by + BOMB_RADIUS + 1)):
                p = (x, y)
                if self._line_clear(p, base, unknown_blocks=False):
                    out.append(p)
        return out

    def _nearest_safe(self, loc: tuple[int, int], danger: set[tuple[int, int]], max_depth: int = 8) -> tuple[int, int] | None:
        if loc not in danger:
            return loc
        q = deque([(loc, 0)])
        seen = {loc}
        while q:
            pos, depth = q.popleft()
            if pos not in danger:
                return pos
            if depth >= max_depth:
                continue
            for d, nb in self._dir_neighbours(pos):
                if nb in seen or not self._can_step(pos, d, True):
                    continue
                seen.add(nb)
                q.append((nb, depth + 1))
        return None

    def _action_to(self, loc: tuple[int, int], direction: int, target: tuple[int, int] | None, mask: list[int], avoid: set[tuple[int, int]]) -> int:
        if target is None or target == loc:
            return self._explore(loc, direction, mask, avoid)
        path = self._path(loc, target, avoid)
        if not path or len(path) < 2:
            return self._explore(loc, direction, mask, avoid)
        nxt = path[1]
        delta = (nxt[0] - loc[0], nxt[1] - loc[1])
        if delta not in DIRS:
            return self._explore(loc, direction, mask, avoid)
        want = DIRS.index(delta)
        if want == direction and _legal(mask, FORWARD):
            return FORWARD
        if want == OPP[direction] and _legal(mask, BACKWARD):
            return BACKWARD
        if (direction - want) % 4 == 1 and _legal(mask, LEFT):
            return LEFT
        if (want - direction) % 4 == 1 and _legal(mask, RIGHT):
            return RIGHT
        return RIGHT if _legal(mask, RIGHT) else LEFT if _legal(mask, LEFT) else STAY

    def _fallback(self, action: int, loc: tuple[int, int], direction: int, mask: list[int], danger: set[tuple[int, int]]) -> int:
        return action if _legal(mask, action) else self._explore(loc, direction, mask, danger)

    def _explore(self, loc: tuple[int, int], direction: int, mask: list[int], avoid: set[tuple[int, int]]) -> int:
        fwd = (loc[0] + DIRS[direction][0], loc[1] + DIRS[direction][1])
        back_dir = OPP[direction]
        back = (loc[0] + DIRS[back_dir][0], loc[1] + DIRS[back_dir][1])
        opts = []
        if _legal(mask, FORWARD) and self._in_bounds(fwd) and fwd not in avoid:
            opts.append((self.visited[fwd], FORWARD))
        if _legal(mask, BACKWARD) and self._in_bounds(back) and back not in avoid:
            opts.append((self.visited[back] + 0.2, BACKWARD))
        if opts:
            return min(opts)[1]
        if _legal(mask, RIGHT):
            return RIGHT
        if _legal(mask, LEFT):
            return LEFT
        return STAY if _legal(mask, STAY) else self._first_legal(mask)

    def _path(self, start: tuple[int, int], goal: tuple[int, int], avoid: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        pq = [(0.0, start)]
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost = {start: 0.0}
        while pq:
            cur_cost, cur = heapq.heappop(pq)
            if cur == goal:
                break
            if cur_cost != cost[cur]:
                continue
            for d, nb in self._dir_neighbours(cur):
                if nb in avoid or not self._can_step(cur, d, True):
                    continue
                edge_cost = 1.0 + (1.2 if self.walls[cur][d] is None else 0.0) + (2.0 if nb not in self.seen else 0.0)
                edge_cost += min(2.0, 0.15 * self.visited[nb])
                new = cur_cost + edge_cost
                if nb not in cost or new < cost[nb]:
                    cost[nb] = new
                    came[nb] = cur
                    heapq.heappush(pq, (new, nb))
        if goal not in came:
            return None
        path = [goal]
        while path[-1] != start:
            parent = came.get(path[-1])
            if parent is None:
                return None
            path.append(parent)
        return list(reversed(path))

    def _dist(self, start: tuple[int, int], goal: tuple[int, int]) -> float | None:
        p = self._path(start, goal, set())
        return len(p) - 1 if p else abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _can_step(self, pos: tuple[int, int], d: int, allow_unknown: bool) -> bool:
        nb = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
        if not self._in_bounds(pos) or not self._in_bounds(nb):
            return False
        a, b = self.walls[pos][d], self.walls[nb][OPP[d]]
        if a is True or b is True:
            return False
        if not allow_unknown and (a is None or b is None):
            return False
        return True

    def _line_clear(self, start: tuple[int, int], end: tuple[int, int], unknown_blocks: bool) -> bool:
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        nx, ny = abs(dx), abs(dy)
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
        x, y, ix, iy = sx, sy, 0, 0
        while ix < nx or iy < ny:
            px, py = x, y
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                x += sign_x; y += sign_y; ix += 1; iy += 1
            elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
                x += sign_x; ix += 1
            else:
                y += sign_y; iy += 1
            step = (x - px, y - py)
            if step in DIRS:
                if self._edge_blocked((px, py), DIRS.index(step), unknown_blocks):
                    return False
            else:
                dh = DIRS.index((1 if x > px else -1, 0))
                dv = DIRS.index((0, 1 if y > py else -1))
                if self._edge_blocked((px, py), dh, unknown_blocks) and self._edge_blocked((px, py), dv, unknown_blocks):
                    return False
        return True

    def _edge_blocked(self, pos: tuple[int, int], d: int, unknown_blocks: bool) -> bool:
        nb = (pos[0] + DIRS[d][0], pos[1] + DIRS[d][1])
        if not self._in_bounds(pos) or not self._in_bounds(nb):
            return True
        a, b = self.walls[pos][d], self.walls[nb][OPP[d]]
        return a is True or b is True or (unknown_blocks and (a is None or b is None))

    @staticmethod
    def _in_bounds(pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < GRID and 0 <= pos[1] < GRID

    @staticmethod
    def _dir_neighbours(pos: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        return [(d, (pos[0] + dx, pos[1] + dy)) for d, (dx, dy) in enumerate(DIRS) if 0 <= pos[0] + dx < GRID and 0 <= pos[1] + dy < GRID]

    @staticmethod
    def _centre_dist(pos: tuple[int, int]) -> float:
        return math.hypot(pos[0] - 7.5, pos[1] - 7.5)

    @staticmethod
    def _first_legal(mask: list[int]) -> int:
        for i, v in enumerate(mask):
            if v:
                return i
        return STAY
