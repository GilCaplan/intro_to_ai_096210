class GringottsController:
    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc

        # Knowledge state
        self.checked_vaults = set()  # Vaults where Harry has already collected
        self.known_safe = {harry_loc}  # Tiles confirmed as safe
        self.unsafe_tiles = set()  # Tiles confirmed as unsafe (trap or dragon)
        self.known_dragons = set()  # Locations of dragons
        self.known_vaults = set()  # Locations of vaults
        self.visited = {harry_loc}  # Tiles Harry has already visited
        self.literals = {}  # Literals for DPLL logic
        self.clauses = []  # CNF clauses for inference

        # Initial observations processing
        self._process_observations(initial_observations)

    def _process_observations(self, observations):
        """
        Process observations and update internal knowledge of safe/unsafe tiles.
        Observations may include:
            - ("dragon", loc): A dragon is known at `loc`.
            - ("vault", loc): A vault is known at `loc`.
            - ("sulfur",): A trap is suspected nearby based on sulfur smell.
        """
        y, x = self.harry_loc  # Current location
        adjacent_tiles = self._get_adjacent_tiles(y, x)  # Adjacent locations

        # Process each observation from the environment
        for observation in observations:
            if observation[0] == "dragon":
                # A specific dragon location is marked as unsafe
                dragon_tile = observation[1]
                self.known_dragons.add(dragon_tile)
                self.unsafe_tiles.add(dragon_tile)

            elif observation[0] == "vault":
                # A specific vault is discovered
                self.known_vaults.add(observation[1])

            elif observation[0] == "sulfur":
                # Adjacent tiles are candidates for traps based on sulfur smell
                for tile in adjacent_tiles:
                    if tile not in self.known_safe and tile not in self.unsafe_tiles:
                        self.unsafe_tiles.add(tile)

        # If no sulfur is present, adjacent tiles are safe
        if not any(obs[0] == "sulfur" for obs in observations):
            for tile in adjacent_tiles:
                if tile not in self.unsafe_tiles:
                    self.known_safe.add(tile)

    def _add_clause(self, clause):
        """
        Add a clause to the CNF knowledge base.
        """
        if clause not in self.clauses:
            self.clauses.append(clause)

    def _simplify_clauses(self, clauses, literal, value):
        """
        Simplify clauses in the knowledge base based on current assignments.
        """
        simplified_clauses = []
        for clause in clauses:
            # If a clause contains the literal with the assigned value, it's satisfied
            if any(var == literal and val == value for var, val in clause):
                continue
            # Otherwise, remove the literal from the clause
            new_clause = [(var, val) for var, val in clause if var != literal]
            simplified_clauses.append(new_clause)
        return simplified_clauses

    def _get_adjacent_tiles(self, y, x):
        """
        Get valid adjacent tiles (that are within map boundaries).
        """
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        adjacent = [
            (y + dy, x + dx)
            for dy, dx in directions
            if 0 <= y + dy < self.map_shape[0] and 0 <= x + dx < self.map_shape[1]
        ]
        return adjacent

    def _is_safe(self, tile):
        """
        Check if a tile is safe to visit.
        """
        return tile in self.known_safe and tile not in self.unsafe_tiles

    def get_next_action(self, observations):
        """
        Decide the next action for Harry dynamically based on the observations.
        """
        # Process the latest observations to update knowledge
        self._process_observations(observations)

        # Step 1: Collect the vault if standing on it
        if self.harry_loc in self.known_vaults and self.harry_loc not in self.checked_vaults:
            self.checked_vaults.add(self.harry_loc)
            return ("collect",)

        # Step 2: Move to the nearest safe, unvisited tile
        adjacent_tiles = self._get_adjacent_tiles(*self.harry_loc)
        for tile in adjacent_tiles:
            if tile not in self.visited and self._is_safe(tile):
                self.visited.add(tile)
                self.harry_loc = tile  # Update Harry's location
                return ("move", tile)

        # Step 3: Destroy unsafe tiles adjacent to Harry
        for tile in adjacent_tiles:
            if tile in self.unsafe_tiles:
                self.unsafe_tiles.remove(tile)  # Mark the tile as no longer unsafe
                self.known_safe.add(tile)  # Once destroyed, it becomes safe
                return ("destroy", tile)

        # Step 4: If no valid moves or destruction opportunities, wait
        return ("wait",)
