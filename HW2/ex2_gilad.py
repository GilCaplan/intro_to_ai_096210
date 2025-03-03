
ids = ['326627791','326178951']

class GringottsController:

    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.initial_observations = initial_observations
        self.knowledge_base = {}
        self.visited = []
        self.sulfurs = set()
        self.i = 0
        self.flag = False
        self.flag_ = False
        self.existed_trap = False
        self.save_location = None
        self.last_move = None
        self.traps = [[0 for _ in range(self.map_shape[1])] for _ in range(self.map_shape[0])]#measuring if there is a dead end for what is known 0 for unkown 1 for sure yes, -1 for surely not
        self.states = [[0 for _ in range(self.map_shape[1])] for _ in range(self.map_shape[0])]#0 for unkown 1 for sure yes, -1 for surely not
        self.counters = [[0 for _ in range(self.map_shape[1])] for _ in range(self.map_shape[0])]#0 for unkown 1 for sure yes, -1 for surely not
        self.dragons = []
        self.check_back = None
        self.number_iteration = 0
        # TODO: fill in
        # What can we do with only initial_observations, we can't explore the map

    def get_next_action(self, observations):
        self.update_kb(observations)
        self.dragons.extend([observation[1] for observation in observations if observation[0] == 'dragon'])
        right_moves = []
        is_sulfur_in_observation = self.sulfur_in_observations(observations)
        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        for move in moves:
            x_new = self.harry_loc[0] + move[0]
            y_new = self.harry_loc[1] + move[1]
            if self.in_bounds(x_new, y_new):#check if there is a trap
                if not self.is_there_dragon(observations, x_new, y_new):
                    if self.states[x_new][y_new] == -1 or (self.states[x_new][y_new] == 0 and not is_sulfur_in_observation):
                        right_moves.append(("move", (x_new, y_new)))
                    if self.states[x_new][y_new] == 1 and (x_new, y_new) not in self.dragons:
                        right_moves.append(("destroy", (x_new, y_new)))
        
        #Go for capturing the vault, first destroying then moving and collecting no matter what
        if not self.flag and not self.flag_:
            for observation in observations:
                if len(self.visited) > 1 and observation[0] == "vault" and (self.counters[observation[1][0]][observation[1][1]] == 0 or
                self.counters[observation[1][0]][observation[1][1]] == 1 and observation[1][0] == self.visited[0][0] and observation[1][1] == self.visited[0][1]):
                    self.flag = True
                    self.save_location = observation[1]
                    if (self.states[observation[1][0]][observation[1][1]] == 1 or
                            (self.states[observation[1][0]][observation[1][1]] == 0 and is_sulfur_in_observation)):
                        self.last_move = ("destroy", observation[1])
                        return "destroy", observation[1]
        if self.flag:
            self.flag = False
            self.flag_ = True
            self.harry_loc = self.save_location
            self.last_move = ("move", self.save_location)
            return "move", self.save_location

        if self.flag_:
            self.flag_ = False
            self.last_move = ("collect", )
            return ("collect", )

        right_moves_ = [
            move for move in right_moves
            if self.counters[move[1][0]][move[1][1]] < 2
        ]

        if len(right_moves_) == 0:
            for move in moves:
                x_new = self.harry_loc[0] + move[0]
                y_new = self.harry_loc[1] + move[1]
                if self.in_bounds(x_new, y_new):
                    if self.traps[x_new][y_new] == 0:
                        right_moves_.append(("destroy", (x_new, y_new)))

        if len(right_moves_) == 0:
            best_index = min(
                range(len(right_moves)),
                key=lambda i: self.counters[right_moves[i][1][0]][right_moves[i][1][1]]
            )
            right_moves = (right_moves[best_index], )
            best_index = 0
        else:
            right_moves = right_moves_
            scored_moves = [(move, self.score_tile(move[1][0], move[1][1])) for move in right_moves]
            best_index = max(range(len(scored_moves)), key=lambda i: (scored_moves[i][1], -self.counters[scored_moves[i][0][1][0]][scored_moves[i][0][1][1]]))
        if right_moves[best_index][0] == "move":
            self.harry_loc = right_moves[best_index][1]
        self.last_move = right_moves[best_index]
        return right_moves[best_index]

    def is_there_dragon(self, observations, tile_x, tile_y):
        for observation in observations:
            if observation[0] == "dragon":
                if tile_x == observation[1][0] and tile_y == observation[1][1]:
                    return True
        return False

    def is_there_vault_in_observation(self, observations):
        locations = []
        for observation in observations:
            if observation[0] == "vault":
                locations.append(observation[1])
        return locations

    def get_scoring(self, actions_right):
        #exploration
        pass


    def update_kb(self, observations):
        is_sulfur_in_observation = self.sulfur_in_observations(observations)
        try:
            self.counters[self.harry_loc[0]][self.harry_loc[1]] += 1
        except:
            print(self.last_move)
            print(self.map_shape)
            print(self.harry_loc)
            print(self.states)
        self.visited.append(self.harry_loc)
        if is_sulfur_in_observation:
            self.sulfurs.add(self.harry_loc)

        if self.last_move is not None and self.last_move[0] == "destroy":
            self.states[self.last_move[1][0]][self.last_move[1][1]] = -1
            self.traps[self.last_move[1][0]][self.last_move[1][1]] = -1

        self.states[self.harry_loc[0]][self.harry_loc[1]] = -1
        self.traps[self.harry_loc[0]][self.harry_loc[1]] = -1

        moves = [(0,1), (0,-1), (1,0), (-1,0)]
        for observation in observations:
            if observation[0] == "dragon":
                self.states[observation[1][0]][observation[1][1]] = 1

        for sulfur_loc in self.sulfurs:
            number_of_unkown_neighbourhood = 0
            location_save = None
            is_exist = False
            for move in moves:
                x_new = sulfur_loc[0] + move[0]
                y_new = sulfur_loc[1] + move[1]
                if self.in_bounds(x_new, y_new):
                    if self.traps[x_new][y_new] == 1:
                        is_exist = True
                    if self.traps[x_new][y_new] != 1:
                        number_of_unkown_neighbourhood += 1
                        location_save = x_new, y_new

            if number_of_unkown_neighbourhood == 1 and not is_exist:
                self.states[location_save[0]][location_save[1]] = 1
                self.traps[location_save[0]][location_save[1]] = 1

        if not is_sulfur_in_observation:
            for move in moves:
                x_new = self.harry_loc[0] + move[0]
                y_new = self.harry_loc[1] + move[1]
                if self.in_bounds(x_new, y_new):
                    self.traps[x_new][y_new] = -1
                    if not 'dragon' in [observation[0] for observation in observations]:
                        self.states[x_new][y_new] = -1
                    if 'dragon' in [observation[0] for observation in observations]:
                        for observation in observations:
                            if not(observation[0] == 'dragon' and observation[1][0] == x_new and observation[1][1] == y_new):
                                self.states[x_new][y_new] = -1


    def in_bounds(self, x, y):    
        return 0 <= x < self.map_shape[0] and 0 <= y < self.map_shape[1]
    def sulfur_in_observations(self, observations):
        return 'sulfur' in [observation[0] for observation in observations]

    def recursive_score(self, x, y, current_depth, depth, visited=None, unexplored=None):
        if current_depth > depth:
            return 0

        if visited is None:
            visited = set()
        if unexplored is None:
            unexplored = set()

        visited.add((x, y))

        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for move in moves:
            x_new, y_new = x + move[0], y + move[1]
            if self.in_bounds(x_new, y_new) and (x_new, y_new) not in visited:
                if self.states[x_new][y_new] == 0:
                    unexplored.add((x_new, y_new))
                self.recursive_score(x_new, y_new, current_depth + 1, depth, visited, unexplored)

        return len(unexplored)

    def score_tile(self, x, y):
        return self.recursive_score(x, y, 0, 2)