from typing import List, Tuple, Set
from collections import deque
ids = ["318898698"]

class GringottsController:
    def __init__(self, map_shape: Tuple[int, int], harry_loc: Tuple[int, int], initial_observations: List[Tuple]):
        self.map_shape = map_shape
        self.harry_loc = harry_loc

        # Knowledge base
        self.known_safe = {harry_loc}
        self.known_dragons = set()
        self.known_vaults = set()
        self.potential_traps = set()
        self.visited = {harry_loc}
        self.checked_vaults = set()

        # Path planning
        self.current_path = []
        self.nearest_vault = None

        # Process initial observations
        self._process_observations(initial_observations)

    def _process_observations(self, observations: List[Tuple]) -> None:
        """Process new observations and update knowledge base"""
        y, x = self.harry_loc
        adjacent = self._get_adjacent_tiles(y, x)

        has_sulfur = False
        for obs in observations:
            if obs[0] == "dragon":
                self.known_dragons.add(obs[1])
            elif obs[0] == "vault":
                self.known_vaults.add(obs[1])
                if not self.nearest_vault:
                    self.nearest_vault = obs[1]
            elif obs[0] == "sulfur":
                has_sulfur = True

        # Mark adjacent tiles as potentially trapped if sulfur is detected
        if has_sulfur:
            for tile in adjacent:
                if (tile not in self.known_safe and
                        tile not in self.known_dragons):
                    self.potential_traps.add(tile)
        else:
            # If no sulfur, adjacent unexplored tiles are safe
            for tile in adjacent:
                if tile in self.potential_traps:
                    self.potential_traps.remove(tile)
                self.known_safe.add(tile)

    def _get_adjacent_tiles(self, y: int, x: int) -> List[Tuple[int, int]]:
        """Get valid adjacent tiles"""
        adjacent = []
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_y, new_x = y + dy, x + dx
            if 0 <= new_y < self.map_shape[0] and 0 <= new_x < self.map_shape[1]:
                adjacent.append((new_y, new_x))
        return adjacent

    def _find_quick_path(self, target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find a quick path to target using BFS"""
        if target == self.harry_loc:
            return []

        queue = deque([(self.harry_loc, [])])
        visited = {self.harry_loc}

        while queue:
            pos, path = queue.popleft()

            if pos == target:
                return path

            for next_pos in self._get_adjacent_tiles(*pos):
                if (next_pos not in visited and
                        next_pos not in self.known_dragons and
                        (next_pos in self.known_safe or next_pos == target)):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return []

    def get_next_action(self, observations: List[Tuple]) -> Tuple:
        """Determine next action with focus on minimizing turns"""
        self._process_observations(observations)

        # Check if at vault
        if self.harry_loc in self.known_vaults and self.harry_loc not in self.checked_vaults:
            self.checked_vaults.add(self.harry_loc)
            return ("collect",)

        # Handle traps blocking path to vault
        adjacent = self._get_adjacent_tiles(*self.harry_loc)
        for adj in adjacent:
            if adj in self.potential_traps:
                # Only destroy trap if it's blocking path to vault
                if self.nearest_vault and (
                        adj == self.nearest_vault or
                        any(v for v in self.known_vaults if
                            manhattan_distance(adj, v) < manhattan_distance(self.harry_loc, v))
                ):
                    return ("destroy", adj)

        # If we have a vault in sight, go there directly
        if self.nearest_vault and self.nearest_vault not in self.checked_vaults:
            path = self._find_quick_path(self.nearest_vault)
            if path:
                next_pos = path[0]
                self.harry_loc = next_pos
                self.visited.add(next_pos)
                return ("move", next_pos)

        # Explore efficiently
        for adj in adjacent:
            if (adj not in self.visited and
                    adj not in self.known_dragons and
                    adj not in self.potential_traps):
                self.harry_loc = adj
                self.visited.add(adj)
                return ("move", adj)

        # If stuck, carefully destroy nearest trap
        for adj in adjacent:
            if adj in self.potential_traps:
                return ("destroy", adj)

        return ("wait",)


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])