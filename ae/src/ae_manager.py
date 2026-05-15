"""Fast rule-based AE agent for TIL-AI 2026.

The environment is a 16x16 grid. The server receives only partial views, so this
agent builds a lightweight internal map online, then plans greedily to missions,
resources, recon tokens, and attack positions near enemy bases.

It deliberately avoids heavy ML inference: AE scoring includes a speed component,
and the action space is small enough that a strong deterministic planner is a
good baseline for Qualifiers.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict, deque
from typing import Any

# Action ids from til_environment.actions.Action
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

# Direction ids from til_environment.types.Direction
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
OPP = (2, 3, 0, 1)

# Observation channels from til_environment.observation.ViewChannel
VISIBLE = 0
WALL_RIGHT = 1
WALL_DOWN = 2
WALL_LEFT = 3
WALL_UP = 4
TILE_EMPTY = 5
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ALLY_AGENT = 9
ENEMY_AGENT = 10
ALLY_BASE = 11
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


def _as_int_pair(value: Any, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        return (int(value[0]), int(value[1]))
    except Exception:
        return default


def _cell_channel(cell: Any, ch: int) -> float:
    try:
        return float(cell[ch])
    except Exception:
        return 0.0


def _legal(mask: list[int], action: int) -> bool:
    return 0 <= action < len(mask) and bool(mask[action])


class AEManager:
    """Stateful planner. A new instance is created when the FastAPI app starts."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Known walls/destructible flags: cell -> [right, down, left, up]
        # value is None if unknown, False open, True blocked.
        self.walls: dict[tuple[int, int], list[bool | None]] = defaultdict(lambda: [None, None, None, None])
        self.destructible: dict[tuple[int, int], list[bool | None]] = defaultdict(lambda: [None, None, None, None])

        # Known dynamic/static objects.
        self.items: dict[tuple[int, int], tuple[str, int]] = {}
        self.enemy_bases: dict[tuple[int, int], float] = {}
        self.enemy_agents: dict[tuple[int, int], int] = {}
        self.bombs: dict[tuple[int, int], tuple[int, int]] = {}  # pos -> (timer_when_seen, step_seen)

        self.seen: set[tuple[int, int]] = set()
        self.visited: dict[tuple[int, int], int] = defaultdict(int)
        self.last_positions: deque[tuple[int, int]] = deque(maxlen=12)
        self.last_step: int = -1
        self.last_loc: tuple[int, int] | None = None
        self.last_bomb_step: int = -99
        self.evade_until: int = -1
        self.rng = random.Random(1337)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def ae(self, observation: dict[str, Any]) -> int:
        step = int(observation.get("step", 0))
        if step == 0 or (self.last_step >= 0 and step < self.last_step):
            self.reset()

        loc = _as_int_pair(observation.get("location", [0, 0]))
        direction = int(observation.get("direction", 0)) % 4
        mask = [int(x) for x in observation.get("action_mask", [1, 1, 1, 1, 1, 0])]

        self._age_bombs(step)
        self._update_from_observation(observation, step)
        self.visited[loc] += 1
        self.last_positions.append(loc)

        self.last_step = step
        self.last_loc = loc

        if not any(mask):
            return STAY
        if _legal(mask, STAY) and (observation.get("frozen_ticks", 0) or mask.count(1) == 1):
            return STAY

        danger_now, danger_soon = self._danger_cells(step)

        # Priority 1: if we are in blast danger, run before doing anything clever.
        if loc in danger_now or (self.evade_until >= step and loc in danger_soon):
            safe = self._nearest_safe_cell(loc, danger_soon)
            action = self._action_towards(loc, direction, safe, mask, avoid=danger_soon)
            return self._fallback(action, loc, direction, mask, danger_soon)

        # Priority 2: bomb high-value nearby targets if we can still escape.
        if _legal(mask, PLACE_BOMB) and self._should_place_bomb(loc, step, danger_soon):
            self.bombs[loc] = (BOMB_TIMER, step)
            self.last_bomb_step = step
            self.evade_until = step + BOMB_TIMER + 1
            return PLACE_BOMB

        target = self._choose_target(loc, step, danger_soon)
        action = self._action_towards(loc, direction, target, mask, avoid=danger_soon)
        return self._fallback(action, loc, direction, mask, danger_soon)

    # ------------------------------------------------------------------
    # Observation decoding
    # ------------------------------------------------------------------
    def _update_from_observation(self, obs: dict[str, Any], step: int) -> None:
        loc = _as_int_pair(obs.get("location", [0, 0]))
        direction = int(obs.get("direction", 0)) % 4
        base_loc = _as_int_pair(obs.get("base_location", [0, 0]))

        self._read_view(obs.get("agent_viewcone", []), loc, direction, step, agent_view=True)
        self._read_base_view(obs.get("base_viewcone", []), base_loc, step)

    def _read_view(
        self,
        view: Any,
        origin: tuple[int, int],
        direction: int,
        step: int,
        *,
        agent_view: bool,
    ) -> None:
        for i, row in enumerate(view or []):
            for j, cell in enumerate(row or []):
                if _cell_channel(cell, VISIBLE) < 0.5:
                    continue
                if agent_view:
                    # Matches til_environment.observation: view_coord = [i-behind, j-left]
                    rel = (i - VIEW_BEHIND, j - VIEW_LEFT)
                    pos = self._view_to_world(origin, direction, rel)
                else:
                    # Radius/base views are square and centered.
                    side = len(view)
                    r = side // 2
                    pos = (origin[0] + i - r, origin[1] + j - r)
                if self._in_bounds(pos):
                    self._record_cell(pos, cell, step)

    def _read_base_view(self, view: Any, base_loc: tuple[int, int], step: int) -> None:
        self._read_view(view, base_loc, 0, step, agent_view=False)

    @staticmethod
    def _view_to_world(origin: tuple[int, int], direction: int, rel: tuple[int, int]) -> tuple[int, int]:
        x, y = origin
        a, b = rel
        if direction == 0:     # RIGHT
            return (x + a, y + b)
        if direction == 1:     # DOWN
            return (x - b, y + a)
        if direction == 2:     # LEFT
            return (x - a, y - b)
        return (x + b, y - a)  # UP

    def _record_cell(self, pos: tuple[int, int], cell: Any, step: int) -> None:
        self.seen.add(pos)

        wall_channels = (WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP)
        destr_channels = (DESTR_RIGHT, DESTR_DOWN, DESTR_LEFT, DESTR_UP)
        for d in range(4):
            blocked = _cell_channel(cell, wall_channels[d]) >= 0.5
            destr = _cell_channel(cell, destr_channels[d]) >= 0.5
            self._set_wall(pos, d, blocked, destr)

        # Collectibles. Only one of these should be present.
        if _cell_channel(cell, TILE_MISSION) >= 0.5:
            self.items[pos] = ("mission", step)
        elif _cell_channel(cell, TILE_RESOURCE) >= 0.5:
            self.items[pos] = ("resource", step)
        elif _cell_channel(cell, TILE_RECON) >= 0.5:
            self.items[pos] = ("recon", step)
        elif _cell_channel(cell, TILE_EMPTY) >= 0.5:
            self.items.pop(pos, None)

        if _cell_channel(cell, ENEMY_BASE) >= 0.5:
            hp = _cell_channel(cell, ENEMY_BASE_HEALTH)
            self.enemy_bases[pos] = hp if hp > 0 else 1.0

        if _cell_channel(cell, ENEMY_AGENT) >= 0.5:
            self.enemy_agents[pos] = step

        # Bombs: timer channels are absolute-ish, but can be 0 if not stamped;
        # use BOMB_TIMER as fallback for newly seen bombs.
        if _cell_channel(cell, ALLY_BOMB) >= 0.5:
            timer = int(_cell_channel(cell, ALLY_BOMB_TIMER) or BOMB_TIMER)
            self.bombs[pos] = (timer, step)
        if _cell_channel(cell, ENEMY_BOMB) >= 0.5:
            timer = int(_cell_channel(cell, ENEMY_BOMB_TIMER) or BOMB_TIMER)
            self.bombs[pos] = (timer, step)

    def _set_wall(self, pos: tuple[int, int], d: int, blocked: bool, destr: bool) -> None:
        if not self._in_bounds(pos):
            return
        self.walls[pos][d] = blocked
        self.destructible[pos][d] = destr if blocked else False

        dx, dy = DIRS[d]
        nb = (pos[0] + dx, pos[1] + dy)
        if self._in_bounds(nb):
            self.walls[nb][OPP[d]] = blocked
            self.destructible[nb][OPP[d]] = destr if blocked else False

    # ------------------------------------------------------------------
    # Bomb danger and bombing decisions
    # ------------------------------------------------------------------
    def _age_bombs(self, step: int) -> None:
        expired = []
        for pos, (timer, seen_step) in self.bombs.items():
            est = timer - max(0, step - seen_step)
            if est <= -1:
                expired.append(pos)
        for pos in expired:
            self.bombs.pop(pos, None)

        # Enemy agents go stale quickly.
        for pos, seen_step in list(self.enemy_agents.items()):
            if step - seen_step > 8:
                self.enemy_agents.pop(pos, None)

    def _danger_cells(self, step: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        danger_now: set[tuple[int, int]] = set()
        danger_soon: set[tuple[int, int]] = set()
        for bpos, (timer, seen_step) in list(self.bombs.items()):
            est = timer - max(0, step - seen_step)
            if est <= 0:
                self.bombs.pop(bpos, None)
                continue
            cells = self._blast_cells(bpos)
            if est <= 2:
                danger_now.update(cells)
            if est <= BOMB_TIMER:
                danger_soon.update(cells)
        return danger_now, danger_soon

    def _blast_cells(self, bpos: tuple[int, int]) -> set[tuple[int, int]]:
        cells = {bpos}
        bx, by = bpos
        for x in range(max(0, bx - BOMB_RADIUS), min(GRID, bx + BOMB_RADIUS + 1)):
            for y in range(max(0, by - BOMB_RADIUS), min(GRID, by + BOMB_RADIUS + 1)):
                p = (x, y)
                if self._has_line_of_effect(bpos, p, unknown_blocks=False):
                    cells.add(p)
        return cells

    def _should_place_bomb(self, loc: tuple[int, int], step: int, danger: set[tuple[int, int]]) -> bool:
        if step - self.last_bomb_step < 5:
            return False

        # Only bomb if we can leave the blast area.
        simulated_blast = self._blast_cells(loc)
        if not self._nearest_safe_cell(loc, danger | simulated_blast, max_depth=6):
            return False

        # Enemy base in blast radius: very valuable.
        for base_pos in self.enemy_bases:
            if base_pos in simulated_blast:
                return True

        # Enemy agent nearby: worthwhile, but don't spam bombs.
        for enemy_pos, seen_step in self.enemy_agents.items():
            if step - seen_step <= 2 and enemy_pos in simulated_blast:
                return True

        # Break a destructible wall directly in front if it is blocking progress
        # and we have not seen a valuable open target.
        for d in range(4):
            if self.walls[loc][d] is True and self.destructible[loc][d] is True:
                return True

        return False

    # ------------------------------------------------------------------
    # Target selection and pathing
    # ------------------------------------------------------------------
    def _choose_target(self, loc: tuple[int, int], step: int, danger: set[tuple[int, int]]) -> tuple[int, int] | None:
        best: tuple[float, tuple[int, int]] | None = None

        # 1. Known collectibles. Mission > resource > recon.
        values = {"mission": 5.0, "resource": 2.0, "recon": 1.0}
        for pos, (kind, seen_step) in list(self.items.items()):
            if not self._in_bounds(pos):
                self.items.pop(pos, None)
                continue
            dist = self._rough_dist(loc, pos)
            if dist is None:
                continue
            age_penalty = max(0, step - seen_step) * 0.04
            revisit_penalty = self.visited[pos] * 0.7
            centre_bonus = 1.0 / (1.0 + self._centre_dist(pos))
            score = values.get(kind, 0.5) * 100.0 - 4.0 * dist - age_penalty - revisit_penalty + 4.0 * centre_bonus
            if pos in danger:
                score -= 500
            if best is None or score > best[0]:
                best = (score, pos)

        # 2. Enemy base attack positions if we have bombs.
        for base_pos, hp in self.enemy_bases.items():
            for pos in self._candidate_bomb_cells(base_pos):
                if pos in danger:
                    continue
                dist = self._rough_dist(loc, pos)
                if dist is None:
                    continue
                score = 230.0 - 5.0 * dist + (1.0 - hp) * 20.0
                if best is None or score > best[0]:
                    best = (score, pos)

        if best is not None:
            return best[1]

        # 3. Exploration frontier: known cells beside unknown cells.
        frontier = self._frontier_cells()
        best_frontier: tuple[float, tuple[int, int]] | None = None
        for pos in frontier:
            if pos in danger:
                continue
            dist = self._rough_dist(loc, pos)
            if dist is None:
                continue
            new_neighbours = sum(1 for nb in self._neighbours(pos) if nb not in self.seen)
            score = 15.0 * new_neighbours - 2.2 * dist - self.visited[pos] + 5.0 / (1.0 + self._centre_dist(pos))
            if best_frontier is None or score > best_frontier[0]:
                best_frontier = (score, pos)
        if best_frontier is not None:
            return best_frontier[1]

        # 4. Last resort: drift toward centre to find respawning high-value cells.
        return (GRID // 2, GRID // 2)

    def _candidate_bomb_cells(self, base_pos: tuple[int, int]) -> list[tuple[int, int]]:
        out = []
        bx, by = base_pos
        for x in range(max(0, bx - BOMB_RADIUS), min(GRID, bx + BOMB_RADIUS + 1)):
            for y in range(max(0, by - BOMB_RADIUS), min(GRID, by + BOMB_RADIUS + 1)):
                p = (x, y)
                if self._has_line_of_effect(p, base_pos, unknown_blocks=False):
                    out.append(p)
        return out

    def _frontier_cells(self) -> list[tuple[int, int]]:
        out = []
        for pos in self.seen:
            if any(nb not in self.seen and self._can_step(pos, d, allow_unknown=True) for d, nb in self._dir_neighbours(pos)):
                out.append(pos)
        return out

    def _nearest_safe_cell(
        self,
        loc: tuple[int, int],
        danger: set[tuple[int, int]],
        max_depth: int = 8,
    ) -> tuple[int, int] | None:
        if loc not in danger:
            return loc
        q = deque([(loc, 0)])
        parent_seen = {loc}
        best = None
        while q:
            pos, depth = q.popleft()
            if pos not in danger:
                best = pos
                break
            if depth >= max_depth:
                continue
            for d, nb in self._dir_neighbours(pos):
                if nb in parent_seen or not self._in_bounds(nb):
                    continue
                if not self._can_step(pos, d, allow_unknown=True):
                    continue
                parent_seen.add(nb)
                q.append((nb, depth + 1))
        return best

    def _action_towards(
        self,
        loc: tuple[int, int],
        direction: int,
        target: tuple[int, int] | None,
        mask: list[int],
        avoid: set[tuple[int, int]],
    ) -> int:
        if target is None or target == loc:
            return self._explore_action(loc, direction, mask, avoid)

        path = self._path(loc, target, avoid=avoid)
        if not path or len(path) < 2:
            return self._explore_action(loc, direction, mask, avoid)

        nxt = path[1]
        dx, dy = nxt[0] - loc[0], nxt[1] - loc[1]
        try:
            desired_dir = DIRS.index((dx, dy))
        except ValueError:
            return self._explore_action(loc, direction, mask, avoid)

        if desired_dir == direction and _legal(mask, FORWARD):
            return FORWARD
        if desired_dir == OPP[direction] and _legal(mask, BACKWARD):
            return BACKWARD

        # Turn toward desired direction. LEFT/RIGHT are rotations, not strafes.
        if (direction - desired_dir) % 4 == 1 and _legal(mask, LEFT):
            return LEFT
        if (desired_dir - direction) % 4 == 1 and _legal(mask, RIGHT):
            return RIGHT

        # If target is behind, pick a rotation that reduces future turns.
        if _legal(mask, RIGHT):
            return RIGHT
        if _legal(mask, LEFT):
            return LEFT
        return STAY

    def _path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        avoid: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        # Dijkstra over 16x16 is cheap. Unknown edges are allowed but penalised.
        pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost: dict[tuple[int, int], float] = {start: 0.0}

        while pq:
            cur_cost, cur = heapq.heappop(pq)
            if cur == goal:
                break
            if cur_cost != cost[cur]:
                continue
            for d, nb in self._dir_neighbours(cur):
                if not self._in_bounds(nb) or nb in avoid:
                    continue
                if not self._can_step(cur, d, allow_unknown=True):
                    continue
                edge_cost = 1.0
                if self.walls[cur][d] is None:
                    edge_cost += 1.4
                if nb not in self.seen:
                    edge_cost += 2.2
                edge_cost += min(2.0, self.visited[nb] * 0.15)
                new_cost = cur_cost + edge_cost
                if nb not in cost or new_cost < cost[nb]:
                    cost[nb] = new_cost
                    came[nb] = cur
                    heapq.heappush(pq, (new_cost, nb))

        if goal not in came:
            return None
        path = [goal]
        while path[-1] != start:
            parent = came.get(path[-1])
            if parent is None:
                return None
            path.append(parent)
        path.reverse()
        return path

    def _rough_dist(self, start: tuple[int, int], goal: tuple[int, int]) -> float | None:
        path = self._path(start, goal, avoid=set())
        if path is None:
            # Manhattan fallback if the map is still sparse.
            return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        return len(path) - 1

    # ------------------------------------------------------------------
    # Low-level movement / map helpers
    # ------------------------------------------------------------------
    def _fallback(self, action: int, loc: tuple[int, int], direction: int, mask: list[int], danger: set[tuple[int, int]]) -> int:
        if _legal(mask, action):
            return action
        return self._explore_action(loc, direction, mask, danger)

    def _explore_action(self, loc: tuple[int, int], direction: int, mask: list[int], avoid: set[tuple[int, int]]) -> int:
        # Prefer a legal forward/backward move that is not dangerous and not a
        # tight loop; otherwise rotate.
        fwd = (loc[0] + DIRS[direction][0], loc[1] + DIRS[direction][1])
        back_dir = OPP[direction]
        back = (loc[0] + DIRS[back_dir][0], loc[1] + DIRS[back_dir][1])

        candidates = []
        if _legal(mask, FORWARD) and self._in_bounds(fwd) and fwd not in avoid:
            candidates.append((self.visited[fwd], FORWARD))
        if _legal(mask, BACKWARD) and self._in_bounds(back) and back not in avoid:
            candidates.append((self.visited[back] + 0.2, BACKWARD))
        if candidates:
            candidates.sort()
            return candidates[0][1]

        # Rotate toward the side with more unseen space.
        left_dir = (direction + 3) % 4
        right_dir = (direction + 1) % 4
        left_cell = (loc[0] + DIRS[left_dir][0], loc[1] + DIRS[left_dir][1])
        right_cell = (loc[0] + DIRS[right_dir][0], loc[1] + DIRS[right_dir][1])
        left_score = 1 if left_cell not in self.seen else 0
        right_score = 1 if right_cell not in self.seen else 0

        if right_score >= left_score and _legal(mask, RIGHT):
            return RIGHT
        if _legal(mask, LEFT):
            return LEFT
        if _legal(mask, STAY):
            return STAY
        for a in range(len(mask)):
            if _legal(mask, a):
                return a
        return STAY

    def _can_step(self, pos: tuple[int, int], d: int, *, allow_unknown: bool) -> bool:
        if not self._in_bounds(pos):
            return False
        dx, dy = DIRS[d]
        nb = (pos[0] + dx, pos[1] + dy)
        if not self._in_bounds(nb):
            return False
        a = self.walls[pos][d]
        b = self.walls[nb][OPP[d]]
        if a is True or b is True:
            return False
        if not allow_unknown and (a is None or b is None):
            return False
        return True

    def _has_line_of_effect(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        unknown_blocks: bool,
    ) -> bool:
        # Approximate the environment's supercover LOS for bomb blast checks.
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0

        x, y = sx, sy
        ix = iy = 0
        while ix < nx or iy < ny:
            px, py = x, y
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                x += sign_x
                y += sign_y
                ix += 1
                iy += 1
            elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
                x += sign_x
                ix += 1
            else:
                y += sign_y
                iy += 1

            step_dir = self._dir_from_delta(x - px, y - py)
            if step_dir is not None:
                if self._edge_blocked((px, py), step_dir, unknown_blocks=unknown_blocks):
                    return False
            else:
                # Diagonal: if both an x-first and a y-first crossing are blocked,
                # treat the diagonal as blocked.
                dx1 = 0 if x == px else (1 if x > px else -1)
                dy1 = 0 if y == py else (1 if y > py else -1)
                d_h = self._dir_from_delta(dx1, 0)
                d_v = self._dir_from_delta(0, dy1)
                h_block = d_h is not None and self._edge_blocked((px, py), d_h, unknown_blocks=unknown_blocks)
                v_block = d_v is not None and self._edge_blocked((px, py), d_v, unknown_blocks=unknown_blocks)
                if h_block and v_block:
                    return False
        return True

    def _edge_blocked(self, pos: tuple[int, int], d: int, *, unknown_blocks: bool) -> bool:
        dx, dy = DIRS[d]
        nb = (pos[0] + dx, pos[1] + dy)
        if not self._in_bounds(pos) or not self._in_bounds(nb):
            return True
        a = self.walls[pos][d]
        b = self.walls[nb][OPP[d]]
        if a is True or b is True:
            return True
        if unknown_blocks and (a is None or b is None):
            return True
        return False

    @staticmethod
    def _dir_from_delta(dx: int, dy: int) -> int | None:
        try:
            return DIRS.index((dx, dy))
        except ValueError:
            return None

    @staticmethod
    def _in_bounds(pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < GRID and 0 <= pos[1] < GRID

    @staticmethod
    def _neighbours(pos: tuple[int, int]) -> list[tuple[int, int]]:
        return [(pos[0] + dx, pos[1] + dy) for dx, dy in DIRS if 0 <= pos[0] + dx < GRID and 0 <= pos[1] + dy < GRID]

    @staticmethod
    def _dir_neighbours(pos: tuple[int, int]) -> list[tuple[int, tuple[int, int]]]:
        return [(d, (pos[0] + dx, pos[1] + dy)) for d, (dx, dy) in enumerate(DIRS)]

    @staticmethod
    def _centre_dist(pos: tuple[int, int]) -> float:
        return math.hypot(pos[0] - (GRID - 1) / 2.0, pos[1] - (GRID - 1) / 2.0)
