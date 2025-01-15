from collections import deque
import heapq
ids = ['208237479', '208237479']

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
        def zone_discovered(group):
            z = 0
            for (r, c) in group:
                if (r, c) not in self.visited and any(
                        adj not in self.visited for adj in self._get_adjacent_tiles(r, c)):
                    z += 1
            return z

        amounts_max, chosen = 0, None
        for zone_name, zone_coords in self.zones.items():
            cnt = zone_discovered(zone_coords)
            if cnt > amounts_max:
                amounts_max = cnt
                chosen = zone_name

        if not chosen:  # If no unexplored zones found
            return "wait",

        # Find the nearest unvisited coordinate in the chosen zone
        best_coord = None
        min_path_len = float('inf')
        for coord in self.zones[chosen]:
            if coord not in self.visited:
                path = self._astar(self.harry_loc, coord)
                if path and len(path) < min_path_len:
                    min_path_len = len(path)
                    best_coord = coord

        if not best_coord:  # If no reachable coordinates found
            return "wait",

        move = self._astar(self.harry_loc, best_coord)[0]
        if move in self.potential_traps:
            return 'destroy', move
        return 'move', move


    def get_next_action(self, observations):
        """Determine next action with focus on minimizing turns."""
        self._process_observations(observations)
        if self.harry_loc in self.known_vaults and self.harry_loc not in self.checked_vaults:
            self.checked_vaults.add(self.harry_loc)
            return ("collect",)

        nearest_vault, path = self._find_nearest_vault()
        if path:
            next_move = path[0]
            if next_move not in self.potential_traps.union(self.known_dragons):
                self.harry_loc = next_move
                self.visited.add(next_move)
                self.path.append(next_move)
                return "move", next_move

        adjacent = sorted(self._get_adjacent_tiles(*self.harry_loc),
                          key=lambda loc: loc not in
                                          (self.known_vaults - self.checked_vaults).union(self.known_dragons).union(
                                              self.visited))

        for adj in adjacent:
            if adj in self.potential_traps:
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
                adjacent = self._get_adjacent_tiles(*current_pos)
                for adj in adjacent:
                    if adj not in self.visited.union(self.known_dragons.union(self.potential_traps)):
                        path_to_pos = self._astar(self.harry_loc, current_pos)
                        if path_to_pos:
                            next_move = path_to_pos[0]
                            self.harry_loc = next_move
                            self.path.append(next_move)
                            return "move", next_move
        return self.discovered_amounts()
