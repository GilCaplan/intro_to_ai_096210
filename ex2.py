import heapq
ids = ['337604821', '32622390']

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.known_safe = set()
        self.known_dragons = set()
        self.known_vaults = set()
        self.potential_traps = set()
        self.visited = set()
        self.visited_twice = set()
        self.path = []
        self.checked_vaults = set()
        rows, cols = map_shape
        mid_row, mid_col = rows // 2, cols // 2
        self.zones = {
            "upper_left": set((r, c) for r in range(0, mid_row) for c in range(0, mid_col)),
            "upper_right": set((r, c) for r in range(0, mid_row) for c in range(mid_col, cols)),
            "bottom_left": set((r, c) for r in range(mid_row, rows) for c in range(0, mid_col)),
            "bottom_right": set((r, c) for r in range(mid_row, rows) for c in range(mid_col, cols)),
        }

    def _process_observations(self, observations):
        y, x = self.harry_loc
        adjacent = self._get_adjacent_tiles(y, x)
        has_sulfur = False

        for obs in observations:
            if obs[0].lower() ==  "dragon":
                self.known_dragons.add(obs[1])
            elif obs[0].lower() == "vault":
                self.known_vaults.add(obs[1])
            elif obs[0].lower() == "sulfur":
                has_sulfur = True

        if has_sulfur:
            for tile in adjacent:
                if tile not in set(self.known_safe).union(self.known_dragons):
                    self.potential_traps.add(tile)
        else:
            for tile in adjacent:
                self.known_safe.add(tile)
                self.potential_traps.discard(tile)

    def _get_adjacent_tiles(self, y, x):
        return [(y+dy, x+dx) for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= y+dy < self.map_shape[0] and 0 <= x+dx < self.map_shape[1]]

    def _astar(self, start, goal):
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start), 0, start, []))
        visited = set()

        while open_set:
            _, g, current, path = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path

            for adj in self._get_adjacent_tiles(*current):
                if adj not in visited and adj in self.known_safe:
                    new_g = g + 1
                    new_path = path + [adj]
                    heapq.heappush(open_set, (new_g + heuristic(adj), new_g, adj, new_path))
        return []

    def _find_nearest_vault(self):
        nearest_vault = None
        shortest_path = []

        for vault in self.known_vaults - self.checked_vaults:
            path = self._astar(self.harry_loc, vault)
            if not shortest_path or (path and len(path) < len(shortest_path)):
                nearest_vault = vault
                shortest_path = path

        return nearest_vault, shortest_path

    def discovered_amounts(self):
        def zone_potential(group):
            z = 0
            accessible_unvisited = 0
            for (r, c) in group:
                if (r, c) in self.known_dragons:
                    continue
                if (r, c) not in self.visited:
                    neighbors = self._get_adjacent_tiles(r, c)
                    unvisited_neighbors = sum(1 for adj in neighbors if adj not in self.visited)
                    if unvisited_neighbors > 0:
                        z += 1
                        path = self._astar(self.harry_loc, (r, c))
                        if path:
                            accessible_unvisited += 1
            return z + (accessible_unvisited * 2)

        best_score = -1
        best_path = None

        for zone_name, zone_coords in self.zones.items():
            score = zone_potential(zone_coords)
            if score > best_score:
                for coord in zone_coords:
                    if coord not in self.visited and coord not in self.known_dragons:
                        path = self._astar(self.harry_loc, coord)
                        if path:
                            best_score = score
                            best_path = path
                            break
        if not best_path:
            adjacent = self._get_adjacent_tiles(*self.harry_loc)
            for adj in adjacent:
                if adj in self.known_safe and adj not in self.known_dragons:
                    return 'move', adj
            return "wait",

        next_move = best_path[0]
        if next_move in self.potential_traps:
            return 'destroy', next_move

        return 'move', next_move

    def get_next_action(self, observations):
        """Determine next action with focus on minimizing turns."""
        self._process_observations(observations)
        if self.harry_loc in self.known_vaults and self.harry_loc not in self.checked_vaults:
            self.checked_vaults.add(self.harry_loc)
            return ("collect",)

        # Keep the nearest vault logic
        nearest_vault, path = self._find_nearest_vault()
        if path:
            next_move = path[0]
            if next_move not in self.potential_traps.union(self.known_dragons):
                self.harry_loc = next_move
                self.visited.add(next_move)
                self.path.append(next_move)
                return "move", next_move

        # Modified adjacent tile prioritization
        adjacent = sorted(self._get_adjacent_tiles(*self.harry_loc),
                          key=lambda loc: (
                              loc in (self.known_vaults - self.checked_vaults),  # First priority: unvisited vaults
                              loc not in self.visited,  # Second priority: unvisited tiles
                              loc in self.potential_traps,  # Third priority: traps to clear
                              loc not in self.known_dragons  # Last priority: avoid dragons
                          ), reverse=True)  # reverse=True makes True values come first

        for adj in adjacent:
            if adj in self.potential_traps:
                neighbors = self._get_adjacent_tiles(*adj)
                if any(n not in self.visited for n in neighbors):  # Only destroy if it blocks unexplored area
                    self.potential_traps.discard(adj)
                    self.known_safe.add(adj)
                    return "destroy", adj
            if adj not in self.visited.union(self.known_dragons):
                self.harry_loc = adj
                self.visited.add(adj)
                self.path.append(adj)
                return "move", adj

        if len(self.path) > 1:
            for i in range(len(self.path) - 1, -1, -1):
                current_pos = self.path[i]
                position_adjacent = self._get_adjacent_tiles(*current_pos)
                unvisited = [adj for adj in position_adjacent
                             if adj not in self.visited.union(self.known_dragons)]
                if unvisited:
                    path_to_pos = self._astar(self.harry_loc, current_pos)
                    if path_to_pos:
                        next_move = path_to_pos[0]
                        self.harry_loc = next_move
                        self.path.append(next_move)
                        return "move", next_move

        return self.discovered_amounts()