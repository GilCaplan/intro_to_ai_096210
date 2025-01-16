import heapq
from collections import deque

ids = ['337604821', '32622390']


class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.known_safe = {harry_loc}  # Starting position is always safe
        self.known_dragons = set()
        self.known_vaults = set()
        self.potential_traps = set()
        self.visited = {harry_loc}  # Mark starting position as visited
        self.checked_vaults = set()
        # self.path = []

        # Initialize frontier - tiles we know are accessible but haven't visited
        self.frontier = set()
        for adj in self._get_adjacent_tiles(*harry_loc):
            self.frontier.add(adj)

        # Process initial observations
        self._process_observations(initial_observations)

        # Cache for pathfinding
        self.path_cache = {}

        # Keep track of deadends to avoid revisiting
        self.deadends = set()

    def _process_observations(self, observations):
        y, x = self.harry_loc
        adjacent = self._get_adjacent_tiles(y, x)
        has_sulfur = False

        # Process all observations first to get complete information
        for obs in observations:
            if obs[0].lower() == "dragon":
                dragon_loc = obs[1]
                self.known_dragons.add(dragon_loc)
                self.frontier.discard(dragon_loc)
                self.known_safe.discard(dragon_loc)
            elif obs[0].lower() == "vault":
                self.known_vaults.add(obs[1])
            elif obs[0].lower() == "sulfur":
                has_sulfur = True

        # Update knowledge based on sulfur observation
        if has_sulfur:
            for tile in adjacent:
                if tile not in self.known_safe.union(self.known_dragons):
                    self.potential_traps.add(tile)
                    self.frontier.discard(tile)
        else:
            for tile in adjacent:
                if tile not in self.known_dragons:
                    self.known_safe.add(tile)
                    self.potential_traps.discard(tile)
                    if tile not in self.visited:
                        self.frontier.add(tile)

    def _get_adjacent_tiles(self, y, x):
        return [(y + dy, x + dx) for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if 0 <= y + dy < self.map_shape[0] and 0 <= x + dx < self.map_shape[1]]

    def _bfs(self, start, target_set):
        """Modified BFS that returns path to nearest target from set of targets"""
        if not target_set:
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current in target_set:
                return path

            for adj in self._get_adjacent_tiles(*current):
                if adj not in visited and adj in self.known_safe:
                    visited.add(adj)
                    queue.append((adj, path + [adj]))
        return []

    def _evaluate_move(self, pos):
        """Evaluate the utility of moving to a position"""
        if pos in self.deadends:
            return -float('inf')

        score = 0
        # Prefer unvisited positions
        if pos not in self.visited:
            score += 10

        # Prefer positions near unvisited frontiers
        adjacent = self._get_adjacent_tiles(*pos)
        frontier_neighbors = sum(1 for adj in adjacent if adj in self.frontier)
        score += frontier_neighbors * 5

        # Prefer positions near unexplored vaults
        vault_neighbors = sum(1 for adj in adjacent if adj in self.known_vaults - self.checked_vaults)
        score += vault_neighbors * 20

        return score
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

    def get_next_action(self, observations):
        self._process_observations(observations)

        # If we're at a vault, check it
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
                # self.path.append(next_move)
                return "move", next_move

        # Get all possible moves
        adjacent = self._get_adjacent_tiles(*self.harry_loc)

        # First priority: Clear threatening traps
        for adj in adjacent:
            if adj in self.potential_traps:
                # Only destroy if trap blocks exploration
                trap_adjacent = self._get_adjacent_tiles(*adj)
                if any(n in self.frontier for n in trap_adjacent):
                    self.potential_traps.discard(adj)
                    self.known_safe.add(adj)
                    return "destroy", adj

        # Second priority: Move to best unexplored safe tile
        safe_moves = [(self._evaluate_move(adj), adj) for adj in adjacent
                      if adj in self.known_safe and adj not in self.known_dragons]

        if safe_moves:
            _, best_move = max(safe_moves)
            if best_move in self.frontier:
                self.frontier.discard(best_move)
            self.visited.add(best_move)
            self.harry_loc = best_move

            # Check if we've hit a deadend
            next_moves = [adj for adj in self._get_adjacent_tiles(*best_move)
                          if adj in self.known_safe and adj not in self.visited
                          and adj not in self.known_dragons]
            if not next_moves:
                self.deadends.add(best_move)

            return "move", best_move

        # If no good moves available, wait
        return ("wait",)