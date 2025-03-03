import heapq

ids = ['337604821', '32622390']


class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.known_safe = {harry_loc}
        self.known_dragons = set()
        self.known_vaults = set()
        self.potential_traps = set()
        self.visited = {harry_loc}
        self.checked_vaults = set()
        self.path = [harry_loc]
        self.dead_ends = set()
        self.backtrack_counter = 0
        rows, cols = map_shape
        mid_row, mid_col = rows // 2, cols // 2
        self.zones = {
            "upper_left": {(r, c) for r in range(mid_row) for c in range(mid_col)},
            "upper_right": {(r, c) for r in range(mid_row) for c in range(mid_col, cols)},
            "bottom_left": {(r, c) for r in range(mid_row, rows) for c in range(mid_col)},
            "bottom_right": {(r, c) for r in range(mid_row, rows) for c in range(mid_col, cols)}
        }
        self._process_observations(initial_observations)

    def _process_observations(self, observations):
        y, x = self.harry_loc
        adjacent = self._get_adjacent_tiles(y, x)
        has_sulfur = False
        for obs in observations:
            if obs[0].lower() == "dragon":
                dragon_loc = obs[1]
                self.known_dragons.add(dragon_loc)
                self.known_safe.discard(dragon_loc)
                self.visited.discard(dragon_loc)
            elif obs[0].lower() == "vault":
                self.known_vaults.add(obs[1])
            elif obs[0].lower() == "sulfur":
                has_sulfur = True
        if has_sulfur:
            safe_and_dragons = self.known_safe.union(self.known_dragons)
            self.potential_traps.update(tile for tile in adjacent if tile not in safe_and_dragons)
        else:
            self.known_safe.update(adjacent)
            self.potential_traps.difference_update(adjacent)

    def _get_adjacent_tiles(self, y, x):
        return [(y + dy, x + dx) for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= y + dy < self.map_shape[0] and 0 <= x + dx < self.map_shape[1]]

    def _astar(self, start, goal):
        if start == goal: return []

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = [(h(start), 0, start, [])]
        visited = {start}
        while open_set:
            _, g, current, path = heapq.heappop(open_set)
            if current == goal: return path
            for adj in self._get_adjacent_tiles(*current):
                if adj not in visited and adj in self.known_safe and adj not in self.known_dragons:
                    visited.add(adj)
                    new_path = path + [adj]
                    heapq.heappush(open_set, (g + 1 + h(adj), g + 1, adj, new_path))
        return []

    def _find_nearest_vault(self):
        unvisited_vaults = self.known_vaults - self.checked_vaults
        if not unvisited_vaults: return None, []
        best_path, best_vault = None, None
        min_len = float('inf')
        for vault in unvisited_vaults:
            if abs(vault[0] - self.harry_loc[0]) + abs(vault[1] - self.harry_loc[1]) < min_len:
                path = self._astar(self.harry_loc, vault)
                if path and len(path) < min_len:
                    min_len = len(path)
                    best_path = path
                    best_vault = vault
        return best_vault, best_path

    def get_next_action(self, observations):
        self._process_observations(observations)
        if self.harry_loc in self.known_vaults and self.harry_loc not in self.checked_vaults:
            self.checked_vaults.add(self.harry_loc)
            return ("collect",)

        _, vault_path = self._find_nearest_vault()
        if vault_path:
            next_move = vault_path[0]
            if next_move in self.potential_traps:
                self.potential_traps.discard(next_move)
                self.known_safe.add(next_move)
                return "destroy", next_move
            if next_move not in self.known_dragons:
                self.harry_loc = next_move
                self.visited.add(next_move)
                self.path.append(next_move)
                return "move", next_move

        adjacent = sorted(self._get_adjacent_tiles(*self.harry_loc),
                          key=lambda loc: (loc in (self.known_vaults - self.checked_vaults),
                                           loc not in self.visited,
                                           loc in self.potential_traps,
                                           loc not in self.known_dragons,
                                           loc not in self.dead_ends),
                          reverse=True)

        for adj in adjacent:
            if adj in self.potential_traps:
                neighbors = self._get_adjacent_tiles(*adj)
                if any(n not in self.visited for n in neighbors):
                    self.potential_traps.discard(adj)
                    self.known_safe.add(adj)
                    return "destroy", adj
            if adj not in self.visited.union(self.known_dragons):
                self.harry_loc = adj
                self.visited.add(adj)
                self.path.append(adj)
                return "move", adj

        self.backtrack_counter = (self.backtrack_counter + 1) % 11

        for pos in reversed(self.path):
            if pos not in self.dead_ends:
                unvisited = [adj for adj in self._get_adjacent_tiles(*pos)
                             if adj not in self.visited.union(self.known_dragons)]
                if unvisited:
                    path = self._astar(self.harry_loc, pos)
                    if path:
                        next_move = path[0]
                        self.harry_loc = next_move
                        self.path.append(next_move)
                        return "move", next_move

        unvisited_target = self._find_nearest_unvisited()
        if unvisited_target:
            if unvisited_target in self.potential_traps:
                return "destroy", unvisited_target
            self.harry_loc = unvisited_target
            self.path.append(unvisited_target)
            return "move", unvisited_target

    def _find_nearest_unvisited(self):
        min_dist = float('inf')
        best_target = None

        curr_y, curr_x = self.harry_loc
        mid_y, mid_x = self.map_shape[0] // 2, self.map_shape[1] // 2

        for y in range(self.map_shape[0]):
            for x in range(self.map_shape[1]):
                if (y, x) not in self.visited and (y, x) not in self.known_dragons:
                    same_quadrant = (y > mid_y) == (curr_y > mid_y) and (x > mid_x) == (curr_x > mid_x)
                    dist = abs(y - curr_y) + abs(x - curr_x)
                    if same_quadrant:
                        dist *= 0.8

                    if dist < min_dist:
                        path = self._astar(self.harry_loc, (y, x))
                        if path:
                            min_dist = dist
                            best_target = path[0]

        return best_target if best_target else None

