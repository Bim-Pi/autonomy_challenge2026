"""
agent.py -- autonomy team application challenge

APPROACH
--------
Metaphor: a chicken with a flashlight crossing a road it has never seen. It never
steps onto ground it hasn't personally checked, it edges toward the far side one lit
patch at a time, and if the road changes under it, it just deals with that on the fly
instead of freezing up.

UNKNOWN SPACE -- treated as BLOCKED. A cell is only ever driveable once this agent has
personally scanned it and found it clear. This is the cautious choice named in the
brief: no betting on faith that fog conceals nothing. The cost is that early on, before
much has been seen, there usually is no route to the real goal yet -- so the planner
instead drives toward the best "frontier" cell: a cell we know is free, that borders
cells we've never seen, chosen to minimize (cost to reach it) + (straight-line distance
from it to the goal). That's a cheap stand-in for "which unopened door gets me closest
to where I'm going", and it's what turns blind exploration into forward progress.

MEMORY -- there's no explicit forgetting timer. Every tick, the *entire* visible scan
window is written into the map, overwriting whatever was remembered for those cells
before. So if an event removes a wall the agent previously logged as blocked, the very
next time that patch of ground is back in the scan window, the stale memory is simply
replaced by what's actually there now -- freshness is "the last thing I saw is true."

REPLANNING -- not literally every tick (a fixed periodic check is cheap here, but re-deriving
inflation from scratch every single tick is needless work for no behavioral gain most ticks).
A replan is triggered when: (a) there's no path yet, (b) the path we're currently following now
crosses a cell the map disagrees with (something appeared, or we're still routing around
something that's now gone), or (c) a short fixed number of ticks has passed, so newly-seen
ground gets a chance to reveal a shortcut or a better exploration target. That cadence is
deliberately short (every couple ticks, not tens) -- empirically, reconsidering the current best
frontier frequently corrects early, imperfect exploration choices before they cost much distance,
and the wall-clock budget has enormous headroom (single-digit seconds used out of 30s allowed)
to afford it. A* is fast enough on this grid to just run synchronously inside step() -- there's
no separate "thinking" tick that skips a move; the very tick that triggers a replan still
returns a velocity.

FOLLOWER -- see the FOLLOWER paragraph below, after mapping/margin, since the follower's
safety properties depend directly on the zero-margin inflation decision explained there.

OBSTACLE MARGIN -- inflated by exactly robot_radius (no extra slack): some passable gaps in
these scenarios are sized *exactly* for the robot, and any additional buffer closes them
outright, sending the planner off on a doomed search for a bypass that doesn't exist. Distance
from a blocked cell is measured as true rectangle-to-rectangle clearance, not cell-center
distance -- two cells `dx` apart have `max(0, |dx|-1) * res` metres of real gap between their
edges. Cell-center distance alone underestimates exactly this near a corner, which is how an
early version of this agent clipped a wall's corner even though every path *waypoint* it had
picked was individually safe. The arena's outer boundary gets the same treatment as a wall.

A cell can be *legally* safe (clearance >= robot_radius) while still having zero slack left --
e.g. in a doorway a bit wider than the robot, the outermost still-legal row sits at *exactly*
robot_radius of clearance, with nothing left to absorb real acceleration-limited overshoot. A
second, wider "soft" tier flags cells that are legal but tight and makes routing through them
cost extra in A* (not forbidden -- just deprioritized), which pulls the path toward the middle
of a wide-enough gap while still allowing the razor's-edge route on scenarios (like the exact-
width corridor above) where there genuinely is no alternative.

FOLLOWER -- a lookahead ("pure pursuit") tracker, with its aim point checked for line-of-sight
safety (a "smoothed" straight-line shortcut between two safe waypoints is not automatically
itself safe -- it can clip a corner between them) and speed governed by real kinematics rather
than an ad hoc turn-angle multiplier: it scans ahead along the path, and for every upcoming
turn, works backwards using standard braking-distance math (at a_max) to the fastest speed
allowed *right now* such that the robot can actually be down to a safe speed by the time it
gets there. With zero inflation margin, arriving at a sharp corner still carrying speed is a
collision, not just an inefficiency -- this is what keeps that from happening. Waypoint
advancement tracks the closest point on the path ahead of the robot (not just "am I near the
next one"), since the lookahead target can otherwise put the true trajectory further along the
path than a naive single-step check would notice. Speed also eases down over the final ~1m so
the robot settles inside goal_tol instead of overshooting it.

EXPLORATION-EXHAUSTION FALLBACK -- if the known-free region has been fully explored and no
path to the goal *and* no frontier cell remain (a closed room, or a wall with genuinely no
gap yet), the agent does not simply idle wherever the last path left it. It explicitly drives
to and holds at whichever known-reachable cell is closest to the goal. Without this, a robot
that finishes mapping a dead-end can end up parked in a random corner that never has the spot
where a wall later opens inside its sense window, and it would wait out the entire tick budget
never knowing anything changed. Parking at the closest reachable point to the goal is also, not
coincidentally, usually the most useful place to be watching from when something does open.

WHAT'S MISSING / SIMPLIFICATIONS -- no explicit pose-noise filtering; kept simple since 2cm
sigma is small next to a 10cm cell and the zero-margin inflation is compensated for by the
kinematic braking behavior rather than a position filter. The frontier search is a single
best-pick per replan rather than a full information-gain explorer. Path smoothing is entirely
the lookahead follower's job -- the raw A* path itself can still be grid-jagged. Deterministic
given a fixed seed (no agent-side randomness).
"""

import math, heapq

class Agent:
  def __init__(self, cfg:dict):
    self.res = cfg["resolution"]
    self.W = round(cfg["width_m"] / self.res)
    self.H = round(cfg["height_m"] / self.res)
    self.r = cfg["robot_radius"]
    self.v_max = cfg["v_max"]
    self.a_max = cfg["a_max"]
    self.goal = cfg["goal"]
    self.goal_tol = cfg["goal_tol"]

    # Two different clearance needs, kept deliberately separate: a corridor can be sized *exactly*
    # for the robot (no slack at all -- inflating by even a couple cm there closes it entirely and
    # sends the planner off on a doomed search for a nonexistent bypass), while a sharp turn lets
    # the real, acceleration-limited robot drift measurably wide of the straight path we validated
    # (bounded speed near turns helps, but can't eliminate this -- momentum is real). So: the hard
    # planning cutoff stays at exactly robot_radius (no bypass-search regression), and a small extra
    # buffer only pads how far that cutoff searches/what counts as "close enough to worry about",
    # not what a straight, low-curvature corridor needs to leave open.
    self.buffer = 0.0   # the corridor test has EXACTLY robot_radius of real clearance and nothing more --
                         # confirmed by hand against its own geometry, so any positive buffer here closes
                         # it outright. Drift protection instead lives entirely in the follower (below):
                         # tight lookahead + aggressive turn slowdown, not a bigger inflation margin.
    cutoff = self.r + self.buffer
    self.infl_cells = max(1, math.ceil(cutoff / self.res)) + 1
    # offsets (dx, dy) from a blocked cell that are within `cutoff` of it, measured as true
    # rectangle-to-rectangle clearance (not cell-center distance) so a path can't cut a corner:
    # two axis-aligned cells `dx` apart have `max(0, |dx| - 1) * res` metres of real gap between
    # their edges, same for dy -- cell-center distance alone underestimates exactly this, letting a
    # diagonal path graze a wall's corner closer than robot_radius actually allows.
    self._infl_offsets = []
    for dx in range(-self.infl_cells, self.infl_cells + 1):
      for dy in range(-self.infl_cells, self.infl_cells + 1):
        gap = math.hypot(max(0, abs(dx) - 1) * self.res, max(0, abs(dy) - 1) * self.res)
        if gap < cutoff: self._infl_offsets.append((dx, dy))

    # A cell can be legally safe (gap >= robot_radius) while still having exactly zero slack --
    # e.g. a doorway a bit wider than the robot lets A* route along its outermost still-legal row,
    # which sits at *exactly* robot_radius of clearance with nothing left to absorb the real
    # robot's acceleration-limited overshoot. So: a second, wider "soft" tier flags cells that are
    # legal but tight (clearance < robot_radius + soft_pad), and _neighbors (below) makes routing
    # through them cost extra -- steering the path toward the middle of a wide-enough gap, while
    # still allowing them when, as in the exact-width corridor test, there's truly no alternative.
    self.soft_pad = 0.15
    soft_cutoff = self.r + self.soft_pad
    soft_cells = max(1, math.ceil(soft_cutoff / self.res)) + 1
    self._soft_offsets = []
    for dx in range(-soft_cells, soft_cells + 1):
      for dy in range(-soft_cells, soft_cells + 1):
        gap = math.hypot(max(0, abs(dx) - 1) * self.res, max(0, abs(dy) - 1) * self.res)
        if gap < soft_cutoff: self._soft_offsets.append((dx, dy))
    self.soft_bias_cost = 0.6 * self.res

    self.known:dict[tuple[int, int], bool] = {}   # cell -> True(blocked)/False(free); absent = never seen
    self.blocked_known:set[tuple[int, int]] = set()
    self.free_known:set[tuple[int, int]] = set()
    self.inflated:set[tuple[int, int]] = set()   # known-free cells too close to a wall to route through
    self.danger:set[tuple[int, int]] = set()      # known-free, legal, but tight -- avoid when an alternative exists

    self.path:list[tuple[float, float]] = []
    self.path_kind:str|None = None   # "goal" once a real path to the goal is known, else "frontier"/"wait"
    self.frontier_target:tuple[int, int]|None = None   # the cell we're currently committed to exploring toward
    self.wp_idx = 0
    self.ticks_since_replan = 10 ** 9   # force a replan on the very first tick
    self.REPLAN_EVERY = 2

    self.lookahead = max(0.30, self.r * 1.2)
    self.advance_eps = max(self.res * 1.5, 0.05)

  # ---------- coordinate helpers ----------
  def _to_cell(self, pt:tuple[float, float]) -> tuple[int, int]: return (math.floor(pt[0] / self.res), math.floor(pt[1] / self.res))
  def _cell_center(self, c:tuple[int, int]) -> tuple[float, float]: return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)
  def _in_bounds(self, c:tuple[int, int]) -> bool: return 0 <= c[0] < self.W and 0 <= c[1] < self.H

  def _traversable(self, c:tuple[int, int]) -> bool:
    cx, cy = c
    if not self._in_bounds(c): return False
    # the arena boundary is a wall too -- keep the same clearance off it as off any scanned obstacle
    if cx < self.infl_cells or cy < self.infl_cells or cx >= self.W - self.infl_cells or cy >= self.H - self.infl_cells: return False
    return self.known.get(c) is False and c not in self.inflated

  # ---------- mapping ----------
  def _update_map(self, scan:tuple[int, int, list[str]]):
    cx0, cy0, rows = scan
    for j, row in enumerate(rows):
      cy = cy0 + j
      if not (0 <= cy < self.H): continue
      for i, ch in enumerate(row):
        cx = cx0 + i
        if not (0 <= cx < self.W): continue
        cell, blocked = (cx, cy), ch == "#"
        prev = self.known.get(cell)
        if prev is blocked: continue   # nothing changed, skip the set churn
        self.known[cell] = blocked
        (self.free_known, self.blocked_known)[blocked].add(cell)
        (self.blocked_known, self.free_known)[blocked].discard(cell)

  def _recompute_inflation(self):
    self.inflated, self.danger = set(), set()
    for bx, by in self.blocked_known:
      for dx, dy in self._infl_offsets:
        c = (bx + dx, by + dy)
        if self.known.get(c) is False: self.inflated.add(c)
      for dx, dy in self._soft_offsets:
        c = (bx + dx, by + dy)
        if self.known.get(c) is False: self.danger.add(c)

  def _nearest_free(self, c:tuple[int, int]) -> tuple[int, int]|None:
    if self._traversable(c): return c
    cx, cy = c
    for rad in range(1, 60):   # expanding ring search
      for dx in range(-rad, rad + 1):
        for dy in (-rad, rad):
          cand = (cx + dx, cy + dy)
          if self._traversable(cand): return cand
      for dy in range(-rad + 1, rad):
        for dx in (-rad, rad):
          cand = (cx + dx, cy + dy)
          if self._traversable(cand): return cand
    return None

  # ---------- planning ----------
  def _neighbors(self, c:tuple[int, int]):
    cx, cy = c
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
      nb = (cx + dx, cy + dy)
      if not self._traversable(nb): continue
      if dx and dy:
        if not (self._traversable((cx + dx, cy)) and self._traversable((cx, cy + dy))): continue   # no slipping between two corner-touching blocks
        cost = self.res * math.sqrt(2)
      else:
        cost = self.res
      if nb in self.danger: cost += self.soft_bias_cost   # legal but tight -- deprioritize, don't forbid
      yield nb, cost

  def _heuristic(self, a:tuple[int, int], b:tuple[int, int]) -> float: return math.hypot((a[0] - b[0]) * self.res, (a[1] - b[1]) * self.res)

  def _astar(self, start:tuple[int, int], goal:tuple[int, int]) -> list[tuple[int, int]]|None:
    frontier = [(0.0, start)]
    came_from, g = {start: None}, {start: 0.0}
    while frontier:
      _, cur = heapq.heappop(frontier)
      if cur == goal:
        path, node = [], cur
        while node is not None:
          path.append(node)
          node = came_from[node]
        return path[::-1]
      for nb, cost in self._neighbors(cur):
        ng = g[cur] + cost
        if nb not in g or ng < g[nb]:
          g[nb] = ng
          came_from[nb] = cur
          heapq.heappush(frontier, (ng + self._heuristic(nb, goal), nb))
    return None

  def _is_frontier(self, c:tuple[int, int]) -> bool:
    cx, cy = c
    return any((cx + dx, cy + dy) not in self.known for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if self._in_bounds((cx + dx, cy + dy)))

  def _best_frontier(self, start:tuple[int, int], goal:tuple[int, int]) -> tuple[tuple[int, int]|None, bool]:
    # Dijkstra over known-free ground. Returns (target, is_frontier). Prefers a frontier cell
    # (known-free, bordering the unknown) minimizing cost-to-reach + straight-line-to-goal, same
    # as before. NEW: if the reachable region has been fully explored and no frontier remains at
    # all (a closed room with no gap, e.g. a wall with no opening yet), that's not "nothing to do"
    # -- it falls back to the single known-reachable cell closest to the goal, so the robot parks
    # itself at the most useful vantage point instead of wherever exploration happened to end.
    # Without this, a robot that maps a whole dead-end room can end up idling in a random corner
    # that never sees the spot where a wall later opens, and waits out the clock forever.
    dist, visited, best_frontier, best_any = {start: 0.0}, set(), None, None
    pq = [(0.0, start)]
    while pq:
      d, c = heapq.heappop(pq)
      if c in visited: continue
      visited.add(c)
      h = self._heuristic(c, goal)
      if best_any is None or h < best_any[0]: best_any = (h, c)
      if self._is_frontier(c):
        score = d + h
        if best_frontier is None or score < best_frontier[0]: best_frontier = (score, c)
      for nb, cost in self._neighbors(c):
        nd = d + cost
        if nb not in dist or nd < dist[nb]:
          dist[nb] = nd
          heapq.heappush(pq, (nd, nb))
    if best_frontier: return best_frontier[1], True
    return (best_any[1], False) if best_any else (None, False)

  def _replan(self, pose:tuple[float, float]):
    self._recompute_inflation()
    start = self._nearest_free(self._to_cell(pose))
    if start is None: return   # nothing known-free nearby yet (shouldn't happen past tick 0); keep old path
    goal_cell = self._to_cell(self.goal)

    cells, kind = None, None
    for target, k in ((goal_cell, "goal"), (self._nearest_free(goal_cell), "goal")):
      if target is not None and self._traversable(target):
        cells = self._astar(start, target)
        if cells:
          kind = k
          break
    if not cells:
      frontier, is_frontier = self._best_frontier(start, goal_cell)
      if frontier is not None:
        cells = self._astar(start, frontier)
        kind = ("frontier" if is_frontier else "wait") if cells else None
    if not cells: return   # no progress possible with current knowledge -- keep following the old path

    pts = [self._cell_center(c) for c in cells]
    pts[0] = pose   # start exactly where we are, not at the cell center, to avoid a tiny jump
    if kind == "goal":
      pts[-1] = self.goal   # finish exactly on the true goal, not the cell center
      self.frontier_target = None   # no longer exploring -- next fallback (if ever needed) should search fresh
    self.path, self.path_kind, self.wp_idx = pts, kind, 0

  def _path_invalid(self) -> bool:
    return any(self.known.get(self._to_cell(pt)) is True for pt in self.path[self.wp_idx:])

  # ---------- following ----------
  def _walk_ahead(self, idx:int, dist:float) -> tuple[tuple[float, float], int]:
    acc, pt = 0.0, self.path[idx]
    while acc < dist and idx < len(self.path) - 1:
      nxt = self.path[idx + 1]
      acc += math.hypot(nxt[0] - pt[0], nxt[1] - pt[1])
      idx, pt = idx + 1, nxt
    return pt, idx

  def _los_clear(self, a:tuple[float, float], b:tuple[float, float]) -> bool:
    # a lookahead target smooths the grid-staircase path into a straight shortcut -- but that
    # shortcut has to be re-checked for safety, since "every waypoint is safe" does not imply
    # "the straight line between two waypoints is safe" (it can clip a corner between them).
    n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / self.res * 2))
    for k in range(n + 1):
      t = k / n
      if not self._traversable(self._to_cell((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))): return False
    return True

  def _safe_lookahead(self, pose:tuple[float, float]) -> tuple[tuple[float, float], int]:
    target, idx = self._walk_ahead(self.wp_idx, self.lookahead)
    while idx > self.wp_idx and not self._los_clear(pose, target):
      idx -= 1
      target = self.path[idx]
    return target, idx

  def _turn_factor_at(self, idx:int) -> float:
    # 1.0 = the path runs straight through this waypoint, 0.0 = it doubles fully back on itself
    if idx <= 0 or idx >= len(self.path) - 1: return 1.0
    ax, ay = self.path[idx - 1]
    bx, by = self.path[idx]
    cx, cy = self.path[idx + 1]
    v1x, v1y, v2x, v2y = bx - ax, by - ay, cx - bx, cy - by
    n1, n2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9: return 1.0
    cosang = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return (1.0 + cosang) / 2.0

  def _brake_cap(self, pose:tuple[float, float]) -> float:
    # with zero inflation margin (see __init__), the robot cannot afford to arrive at a sharp
    # corner still carrying speed -- any acceleration-limited overshoot there is a collision, not
    # just an inefficiency. So: scan ahead along the path, and for every upcoming turn, work out
    # backwards (standard braking-distance kinematics at a_max) the fastest we're allowed to be
    # going *right now* such that we can still be down to a safe speed by the time we reach it.
    cap, idx, dist_acc, pt = self.v_max, self.wp_idx, 0.0, pose
    horizon = self.lookahead * 8
    while idx < len(self.path) - 1 and dist_acc < horizon:
      nxt = self.path[idx + 1]
      dist_acc += math.hypot(nxt[0] - pt[0], nxt[1] - pt[1])
      pt, idx = nxt, idx + 1
      corner_speed = self.v_max * (0.15 + 0.85 * self._turn_factor_at(idx))
      allowed = math.sqrt(max(0.0, corner_speed * corner_speed + 2.0 * self.a_max * dist_acc))
      cap = min(cap, allowed)
    return cap

  def _velocity(self, pose:tuple[float, float]) -> tuple[float, float]:
    if not self.path: return (0.0, 0.0)
    # keep wp_idx at the closest point on the path ahead of it, not just "am I near the current
    # one" -- the line-of-sight-adjusted target can put the real trajectory further along the
    # path than a fixed single-step threshold would notice, leaving wp_idx stale and every safety
    # check downstream reasoning from a position that isn't actually where the robot is anymore.
    while self.wp_idx < len(self.path) - 1:
      here = math.hypot(pose[0] - self.path[self.wp_idx][0], pose[1] - self.path[self.wp_idx][1])
      nxt = math.hypot(pose[0] - self.path[self.wp_idx + 1][0], pose[1] - self.path[self.wp_idx + 1][1])
      if nxt >= here and here >= self.advance_eps: break
      self.wp_idx += 1

    target, idx1 = self._safe_lookahead(pose)
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6: return (0.0, 0.0)
    ux, uy = dx / dist, dy / dist

    turn_cap = self._brake_cap(pose)

    goal_pt = self.path[-1]
    dist_to_goal = math.hypot(pose[0] - goal_pt[0], pose[1] - goal_pt[1])
    ramp = 1.0
    goal_cap = self.v_max if dist_to_goal > ramp else self.v_max * max(0.2, dist_to_goal / ramp)

    speed = min(self.v_max, turn_cap, goal_cap)
    return (ux * speed, uy * speed)

  # ---------- interface ----------
  def step(self, pose:tuple[float, float], scan:tuple[int, int, list[str]]) -> tuple[float, float]:
    self._update_map(scan)
    if not self.path or self._path_invalid() or self.ticks_since_replan >= self.REPLAN_EVERY:
      self._replan(pose)
      self.ticks_since_replan = 0
    else:
      self.ticks_since_replan += 1
    return self._velocity(pose)

  def debug(self) -> dict:
    return {"blocked": self.blocked_known, "free": self.free_known, "path": self.path[self.wp_idx:]}