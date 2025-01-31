from utils import PriorityQueue
from copy import deepcopy
from itertools import product
from collections import defaultdict
import time

ids = ["320546922"]


def order_by_name(a):
    return a[0]


def l1_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def undict_state(state):
    treasure_queue_by_name = PriorityQueue(min, order_by_name)
    marine_queue_by_name = PriorityQueue(min, order_by_name)
    for treasure, treasure_data in state['treasures'].items():
        treasure_queue_by_name.append((treasure, treasure_data))
    for marine, marine_data in state['marine_ships'].items():
        marine_queue_by_name.append((marine, marine_data))
    return treasure_queue_by_name, marine_queue_by_name


def make_initial_state(initial):
    # Sorting the pirates, treasures and marines by name
    treasure_queue, marine_queue = undict_state(initial)
    pirate_queue = PriorityQueue(min, order_by_name)
    for pirate, pirate_data in initial['pirate_ships'].items():
        pirate_queue.append((pirate, pirate_data))

    # Creating lists for the pirates, treasures and marines
    pirates = []

    treasures = []
    treasure_locations = []
    treasure_possible_locations = []
    treasure_probs = []

    marines = []
    marine_indexes = []
    marine_paths = []

    # Getting the base location
    base = pirate_queue.A[0][1][1]['location']

    # Converting the dictionaries to lists
    while pirate_queue:
        pirate, pirate_data = pirate_queue.pop()
        pirates.append(pirate)
    
    while treasure_queue:
        treasure, treasure_data = treasure_queue.pop()
        treasures.append(treasure)
        treasure_locations.append(treasure_data['location'])
        treasure_possible_locations.append(treasure_data['possible_locations'])
        treasure_probs.append(treasure_data['prob_change_location'])
    while marine_queue:
        marine, marine_data = marine_queue.pop()
        marines.append(marine)
        marine_indexes.append(marine_data['index'])
        marine_paths.append(tuple(marine_data['path']))
    
    
    return (base, tuple(treasures), tuple(treasure_locations), tuple(treasure_possible_locations),
            tuple(treasure_probs), tuple(marines), tuple(marine_indexes), tuple(marine_paths), tuple(pirates))


def generate_all_combinations(possible_locations_per_item):
    # Use itertools.product to generate all combinations
    all_combinations = list(product(*possible_locations_per_item))

    # Create a new list with tuples representing actions for each pirate
    result = [tuple(locs) for locs in all_combinations]

    return tuple(result)


class PirateAgent:
    def __init__(self, initial):
        self.start = time.perf_counter()
        self.last_turn = 0
        self.initial = deepcopy(initial)
        self.turns = initial.pop('turns to go')
        data = make_initial_state(initial)
        self.base = data[0]
        self.treasures = data[1]
        self.treasure_initial_locations = data[2]
        self.treasure_possible_locations = data[3]
        self.treasure_probs = data[4]
        self.marines = data[5]
        self.marine_initial_indexes = data[6]
        self.marine_paths = data[7]
        self.pirates = data[8]
        self.height = len(initial["map"])
        self.width = len(initial["map"][0])
        self.map = initial["map"]
        self.prob_cutoff = self.__make_prob_cutoff()
        self.sails_in_loc_dict = self.__all_sails_in_loc()
        self.states, self.environments = self.__make_all_states()
        self.actions_in_state = self.__make_legal_actions()
        self.environment_probability_after_step = self.__make_environment_steps()
        self.states_from_state_and_action = self.__make_states_from_state_and_actions()
        self.policy = self.make_policy()

    def act(self, state):
        turns = state.pop('turns to go')
        treasure_queue, marine_queue = undict_state(state)
        loc = state['pirate_ships'][self.pirates[0]]['location']
        capacity = state['pirate_ships'][self.pirates[0]]['capacity']
        marine_tuple = ()
        treasure_tuple = ()
        while marine_queue:
            marine_tuple += (marine_queue.pop()[1]['index'],)
        while treasure_queue:
            treasure_tuple += (treasure_queue.pop()[1]['location'],)
        state = (loc, marine_tuple, treasure_tuple, capacity)
        action = self.policy[min(turns, self.last_turn)][state][1]
        return action

    def __make_environment_steps(self):
        prob_dict = {}
        for environment in self.environments:
            prob_dict[environment] = self.__environment_steps(environment)
        return prob_dict

    def __make_all_states(self):
        pirate_locs = []
        marines = []
        for i, marine in enumerate(self.marines):
            locs_per_marine = []
            for j in range(len(self.marine_paths[i])):
                locs_per_marine.append(j)
            marines.append(tuple(locs_per_marine))
        marine_indexes = generate_all_combinations(tuple(marines))
        treasure_locs = generate_all_combinations(self.treasure_possible_locations)
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] != 'I':
                    pirate_locs.append((i, j))
        state = (tuple(pirate_locs), marine_indexes, treasure_locs, (0, 1, 2))
        all_states = generate_all_combinations(state)
        all_environments = generate_all_combinations((marine_indexes, treasure_locs))
        return all_states, all_environments

    def make_policy(self):
        policy = [{} for _ in range(self.turns + 1)]
        for state in self.states:
            collision = False
            for i, path_index in enumerate(state[1]):
                if self.marine_paths[i][path_index] == state[0]:
                    collision = True
            if not collision:
                policy[0][state] = (0, 'terminate')
            else:
                policy[0][state] = (-len(self.pirates), 'terminate')
        close_to_timeout = False
        for turn in range(1, self.turns + 1):
            print(f'turn: {turn} time: {time.perf_counter() - self.start}')
            for state in self.states:
                if time.perf_counter() - self.start > 295:
                    close_to_timeout = True
                    break
                collision = False
                for i, path_index in enumerate(state[1]):
                    if self.marine_paths[i][path_index] == state[0]:
                        collision = True
                if not collision:
                    state_reward = 0
                else:
                    state_reward = -len(self.pirates)
                best_action = (0, 'terminate')
                for action in self.actions_in_state[state]:
                    action_value = 0
                    if action[0][0] == 'deposit':
                        action_value = 4 * (2 - state[3]) * len(self.pirates)
                    elif action == 'reset':
                        action_value = -2
                    next_states = self.states_from_state_and_action[state, action]
                    action_future_value = 0
                    for next_state, prob in next_states.items():
                        action_future_value += (policy[turn - 1][next_state][0]) * prob
                    if action_future_value + action_value > best_action[0]:
                        best_action = (action_future_value + action_value, action)
                policy[turn][state] = (best_action[0] + state_reward, best_action[1])
            if close_to_timeout:
                break
            self.last_turn = turn
        return policy

    def __get_adjacent_tiles(self, location):
        valid_sails = []
        islands = []

        def is_valid_move(r, c):
            return 0 <= r < self.height and 0 <= c < self.width

        # Check non-diagonally
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = location[0] + dr, location[1] + dc
            if is_valid_move(new_row, new_col):
                new_loc = self.map[new_row][new_col]
                if new_loc == "S" or new_loc == "B":
                    valid_sails.append((new_row, new_col))
                else:
                    islands.append((new_row, new_col))

        return tuple(valid_sails), tuple(islands)

    def __all_sails_in_loc(self):
        sails_in_loc_dict = {}
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] != 'I':
                    sails_in_loc_dict[(i, j)] = self.__get_adjacent_tiles((i, j))[0]
        return sails_in_loc_dict

    def __make_legal_actions(self):
        all_actions = {}
        for state in self.states:
            actions = ['reset', 'terminate']
            sails = self.sails_in_loc_dict[state[0]]
            for adj in sails:
                sail = []
                for pirate in self.pirates:
                    sail.append(('sail', pirate, adj))
                actions.append(tuple(sail))
            sail = []
            for pirate in self.pirates:
                sail.append(('wait', pirate))
            actions.append(tuple(sail))
            if state[3] > 0:
                for i, loc in enumerate(state[2]):
                    if l1_distance(state[0], loc) == 1:
                        collect = []
                        for pirate in self.pirates:
                            collect.append(('collect', pirate, self.treasures[i]))
                        actions.append(tuple(collect))
            if state[3] < 2 and state[0] == self.base:
                deposit = []
                for pirate in self.pirates:
                    deposit.append(('deposit', pirate))
                actions.append(tuple(deposit))
            all_actions[state] = tuple(actions)
        return all_actions

    def __states_from_action(self, state, action):
        results = defaultdict(lambda: 0)
        if action == 'reset':
            return {(self.base, self.marine_initial_indexes, self.treasure_initial_locations, 2): 1}
        following_env_states = self.environment_probability_after_step[state[1:3]]
        if action[0][0] == 'sail':
            for env in following_env_states:
                collision = False
                for i, path_index in enumerate(env[0]):
                    if self.marine_paths[i][path_index] == action[0][2]:
                        collision = True
                if not collision:
                    results[(action[0][2],) + env[0:2] + (state[3],)] += env[2]
                else:
                    results[(action[0][2],) + env[0:2] + (2,)] += env[2]
        elif action[0][0] == 'wait':
            for env in following_env_states:
                collision = False
                for i, path_index in enumerate(env[0]):
                    if self.marine_paths[i][path_index] == state[0]:
                        collision = True
                if not collision:
                    results[(state[0],) + env[0:2] + (state[3],)] += env[2]
                else:
                    results[(state[0],) + env[0:2] + (2,)] += env[2]
        elif action[0][0] == 'collect':
            for env in following_env_states:
                collision = False
                for i, path_index in enumerate(env[0]):
                    if self.marine_paths[i][path_index] == state[0]:
                        collision = True
                if not collision:
                    results[(state[0],) + env[0:2] + (state[3] - 1,)] += env[2]
                else:
                    results[(state[0],) + env[0:2] + (2,)] += env[2]
        elif action == 'terminate':
            results = {}
        else:
            for env in following_env_states:
                results[(state[0],) + env[0:2] + (2,)] += env[2]
        return results

    def __prob_for_marine(self, path_index, marine_index):
        num_adj = 1
        lower = 0
        upper = 1
        if path_index > 0:
            num_adj += 1
            lower = -1
        if path_index < len(self.marine_paths[marine_index]) - 1:
            num_adj += 1
            upper = 2
        indexes = []
        for i in range(lower, upper):
            indexes.append(((path_index + i,), (), 1 / num_adj))
        return tuple(indexes)

    def __prob_for_treasure(self, treasure_loc, treasure_index):
        prob_dict = {}
        for loc in self.treasure_possible_locations[treasure_index]:
            prob_dict[loc] = 0
        prob_dict[treasure_loc] = (1 - self.treasure_probs[treasure_index])
        for loc in self.treasure_possible_locations[treasure_index]:
            prob_dict[loc] += (
                    self.treasure_probs[treasure_index] / len(self.treasure_possible_locations[treasure_index]))
        return prob_dict

    def __environment_steps(self, environment):
        marine_probabilities = self.__prob_for_marine(environment[0][0], 0)
        for i, path_index in enumerate(environment[0]):
            if i == 0:
                continue
            marine_new_locs = self.__prob_for_marine(path_index, i)
            marine_new_probabilities = []
            for step in marine_probabilities:
                for new_marine in marine_new_locs:
                    marine_new_probabilities.append(((step[0] + new_marine[0]), (), step[2] * new_marine[2]))
            marine_probabilities = marine_new_probabilities
        environment_probabilities = marine_probabilities
        for i, treasure_loc in enumerate(environment[1]):
            treasure_probs = self.__prob_for_treasure(treasure_loc, i)
            environment_new_probabilities = []
            for step in environment_probabilities:
                for new_treasure_loc, prob in treasure_probs.items():
                    if step[2] * prob > self.prob_cutoff:
                        environment_new_probabilities.append((step[0], step[1] + (new_treasure_loc,), step[2] * prob))
                environment_probabilities = environment_new_probabilities
        return tuple(environment_probabilities)

    def __make_prob_cutoff(self):
        if not self.initial['optimal']:
            cutoff = 1
            for path in self.marine_paths:
                if len(path) == 2:
                    cutoff *= 0.5
                elif len(path) >= 3:
                    cutoff *= (1/3)
            highest_prob = 0
            for i, prob in enumerate(self.treasure_probs):
                stay_prob = 1 - prob + (prob/len(self.treasure_possible_locations[i]))
                move_prob = 1 - stay_prob
                cutoff *= max(stay_prob, move_prob)
                highest_prob = max(highest_prob, stay_prob, move_prob)
            cutoff *= (1 - highest_prob)
            cutoff /= highest_prob
            cutoff *= 0.99
            cutoff = min(0.007, cutoff)
        else:
            cutoff = 0
        return cutoff

    def __make_states_from_state_and_actions(self):
        states = {}
        for state in self.states:
            for action in self.actions_in_state[state]:
                states[state, action] = self.__states_from_action(state, action)
        return states


class OptimalPirateAgent(PirateAgent):
    def __init__(self, initial):
        super().__init__(initial)


class InfinitePirateAgent(PirateAgent):
    def __init__(self, initial, gamma):
        initial['turns to go'] = 1
        super().__init__(initial)
        self.gamma = gamma
        self.epsilon = 0.01
        self.policy = self.__make_policy()

    def act(self, state):
        treasure_queue, marine_queue = undict_state(state)
        loc = state['pirate_ships'][self.pirates[0]]['location']
        capacity = state['pirate_ships'][self.pirates[0]]['capacity']
        marine_tuple = ()
        treasure_tuple = ()
        while marine_queue:
            marine_tuple += (marine_queue.pop()[1]['index'],)
        while treasure_queue:
            treasure_tuple += (treasure_queue.pop()[1]['location'],)
        state = (loc, marine_tuple, treasure_tuple, capacity)
        return self.policy[state][1]

    def value(self, state):
        treasure_queue, marine_queue = undict_state(state)
        loc = state['pirate_ships'][self.pirates[0]]['location']
        capacity = state['pirate_ships'][self.pirates[0]]['capacity']
        marine_tuple = ()
        treasure_tuple = ()
        while marine_queue:
            marine_tuple += (marine_queue.pop()[1]['index'],)
        while treasure_queue:
            treasure_tuple += (treasure_queue.pop()[1]['location'],)
        state = (loc, marine_tuple, treasure_tuple, capacity)
        return self.policy[state][0]

    def __make_policy(self):
        policy = [{}]
        for state in self.states:
            collision = False
            for i, path_index in enumerate(state[1]):
                if self.marine_paths[i][path_index] == state[0]:
                    collision = True
            if not collision:
                policy[0][state] = (0, 'terminate')
            else:
                policy[0][state] = (-len(self.pirates), 'terminate')
        converged = False
        turn = 0
        while not converged:
            turn += 1
            policy.append({})
            diff = 0
            close_to_timeout = False
            for state in self.states:
                if time.perf_counter() - self.start > 295:
                    close_to_timeout = True
                    break
                collision = False
                for i, path_index in enumerate(state[1]):
                    if self.marine_paths[i][path_index] == state[0]:
                        collision = True
                if not collision:
                    state_reward = 0
                else:
                    state_reward = -len(self.pirates)
                best_action = (0, 'terminate')
                for action in self.actions_in_state[state]:
                    action_value = 0
                    if action[0][0] == 'deposit':
                        action_value = 4 * (2 - state[3]) * len(self.pirates)
                    elif action == 'reset':
                        action_value = -2
                    next_states = self.states_from_state_and_action[state, action]
                    action_future_value = 0
                    for next_state, prob in next_states.items():
                        action_future_value += self.gamma * (policy[turn - 1][next_state][0]) * prob
                    if action_future_value + action_value > best_action[0]:
                        best_action = (action_future_value + action_value, action)
                policy[turn][state] = (best_action[0] + state_reward, best_action[1])
                diff = max(abs(policy[turn][state][0] - policy[turn - 1][state][0]), diff)
            if close_to_timeout:
                break
            self.last_turn = turn
            if diff < self.epsilon:
                break
        return policy[turn]
