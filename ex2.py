from copy import deepcopy
import heapq

ids = ['6969']

class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        # Initialize observations (literals) for each type of object in the map
        self.dragon_observations = [[0 for _ in range(map_shape[1])]
                                 for _ in range(map_shape[0])]
        self.potential_trap_observations = [[0 for _ in range(map_shape[1])]
                                    for _ in range(map_shape[0])]
        self.vault_observations = [[0 for _ in range(map_shape[1])]
                                    for _ in range(map_shape[0])]

        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.update_observations(initial_observations)

        # Track visited locations and checked vaults
        self.visited = {harry_loc}
        self.checked_vaults = set()

    def update_observations(self, observations):
        for obs in observations:
            if obs[0] == "dragon":
                y, x = obs[1]
                self.dragon_observations[y][x] = 1
            elif obs[0] == "sulfur":
                y, x = self.harry_loc
                for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_y, new_x = y + dy, x + dx
                    if self._is_valid_pos(new_y, new_x):
                        self.potential_trap_observations[new_y][new_x] = 1
            elif obs[0] == "vault":
                y, x = obs[1]
                self.vault_observations[y][x] = 1
    def _is_valid_pos(self, y, x):
        """Check if position is within map bounds"""
        return 0 <= y < self.map_shape[0] and 0 <= x < self.map_shape[1]

    def _get_valid_moves(self):
        """Get list of valid moves from current position"""
        harry_y, harry_x = self.harry_loc
        valid_moves = []
        priority_moves = []
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_y, new_x = harry_y + dy, harry_x + dx
            if not self._is_valid_pos(new_y, new_x):
                continue
            # let's make literals for each object
            dragon = self.dragon_observations[new_y][new_x]
            trap = self.potential_trap_observations[new_y][new_x]
            vault = self.vault_observations[new_y][new_x]
            if ~dragon:
                if trap:
                    if vault:
                        priority_moves.append(("destroy", (new_y, new_x)))
                    valid_moves.append(("destroy", (new_y, new_x)))
                else:
                    if vault:
                        priority_moves.append(("move", (new_y, new_x)))
                    valid_moves.append(("move", (new_y, new_x)))
        if len(priority_moves) > 0:
            return tuple(priority_moves), True
        return tuple(valid_moves), False

    def get_next_action(self, observations):
        # ideas:
        # search map efficiently to get all observations and then go to vaults for treasure
        # or greedy search for vault

        self.update_observations(observations)
        actions, flag = self._get_valid_moves()

        if flag:
            return actions[0]
        for action in actions:
            x, y = self.harry_loc
            # assign score to each action to decide which one to use.
        if actions[0][0] == "move":
            self.visited.add(actions[0][1])
            self.harry_loc = actions[0][1]

        return actions[0]