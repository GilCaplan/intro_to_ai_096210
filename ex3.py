import copy
import itertools
from itertools import product
from value_iteration import *
ids = ["111111111", "222222222"]

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1

class OptimalWizardAgent:
    def __init__(self, initial):
        self.start_state = self.simplify_state(initial)
        self.initial = copy.deepcopy(initial)
        self.map = initial['map']
        self.wizards = initial['wizards']
        self.horcrux = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.num_horcruxes = len(self.horcrux)
        self.all_states = self.compute_states()
        self.rounds = initial['turns_to_go']
        self.cache = ValueIterationCache()
        self.V = self.Value_Iteration()
        self.time = self.rounds

    def simplify_state(self, state):
        return {
            'death_eaters': {key: value['path'][value['index']] for key, value in state['death_eaters'].items()},
            'horcrux': {key: value['location'] for key, value in state['horcrux'].items()},
            'wizards': {key: value['location'] for key, value in state['wizards'].items()}}

    def compute_states(self):
        wiz_locs, death_eaters_locs, horcrux_locs = {}, {}, {}

        # For wizards, make sure to include their initial positions
        for wizard in self.wizards:
            local_wiz = []

            # Then add other possible locations
            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    if self.map[i][j] == 'P':
                        local_wiz.append((i, j))
            wiz_locs[wizard] = local_wiz

        # For death eaters, start with their initial positions
        for de in self.death_eaters:
            death_eaters_locs[de] = [(i, j) for i, j in self.death_eaters[de]['path']]

        # For horcruxes, make sure to include their initial locations
        horcrux_locs = {}
        for hor in self.horcrux:
            horcrux_locs[hor] = []  # Initialize empty list first
            for loc in self.horcrux[hor]['possible_locations']:
                horcrux_locs[hor].append(loc)
            horcrux_locs[hor].append(None)  # Add None for destroyed state

        wizard_combinations = tuple(product(*[wiz_locs[wizard] for wizard in wiz_locs]))
        de_combinations = tuple(product(*[death_eaters_locs[de] for de in death_eaters_locs]))
        horcrux_combinations = tuple(product(*[horcrux_locs[hor] for hor in horcrux_locs]))

        states = []
        # Add initial state first
        if self.start_state not in states:
            states.append(self.start_state)

        # Then generate other states
        for wiz_pos in wizard_combinations:
            for de_pos in de_combinations:
                for hor_pos in horcrux_combinations:
                    state = {'wizards': {}, 'death_eaters': {}, 'horcrux': {}}

                    for i, wizard in enumerate(self.wizards):
                        state['wizards'][wizard] = wiz_pos[i]

                    for i, de in enumerate(self.death_eaters):
                        state['death_eaters'][de] = de_pos[i]

                    for i, hor in enumerate(self.horcrux):
                        state['horcrux'][hor] = hor_pos[i]

                    # Don't add duplicates of the initial state
                    if self.state_to_key(state) != self.state_to_key(self.start_state):
                        states.append(state)

        return states

    def get_actions(self, state):
        actions = []
        wizards = state['wizards']

        def get_move_actions(loc, wiz_name):
            move_actions = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                # Check if new location is within map bounds
                if 0 <= new_loc[0] < len(self.map) and 0 <= new_loc[1] < len(self.map[0]):
                    # Only add move if the destination is 'P' (passable)
                    if self.map[new_loc[0]][new_loc[1]] == 'P':
                        move_actions.append(('move', wiz_name, new_loc))
            return tuple(move_actions)

        def get_destroy_horcrux_actions(loc, wiz_name):
            destroy_actions = []
            horcruxes_here = [hor for hor, h_loc in state['horcrux'].items() if h_loc == loc]
            for hor_name in horcruxes_here:
                destroy_actions.append(('destroy', wiz_name, hor_name))
            return tuple(destroy_actions)

        def get_wait_actions(loc, wiz_name):
            return tuple([('wait', wiz_name, loc)])

        for wizard, wiz_loc in wizards.items():
            wizard_actions = (
                    get_wait_actions(wiz_loc, wizard) +
                    get_move_actions(wiz_loc, wizard) +
                    get_destroy_horcrux_actions(wiz_loc, wizard)
            )
            actions.append(wizard_actions)

        wizard_action_combinations = tuple(itertools.product(*actions))
        all_actions = list(wizard_action_combinations)
        all_actions.append("reset")
        all_actions.append("terminate")

        return tuple(all_actions)

    def state_to_key(self, state):
        wiz = tuple((name, loc) for name, loc in state['wizards'].items())
        de = tuple((name, loc) for name, loc in state['death_eaters'].items())
        hor = tuple((name, loc) for name, loc in state['horcrux'].items())
        return wiz + de + hor

    def calculate_reward(self, state, action):
        reward = 0
        if action == "reset":
            return RESET_REWARD
        if action == "terminate":
            return -1

        if isinstance(action, tuple):
            de_locations = set(state['death_eaters'].values())
            for wizard, loc in state['wizards'].items():
                if loc in de_locations:
                    reward += DEATH_EATER_CATCH_REWARD
            for act in action:
                if act[0] == 'destroy' and act[2] in state['horcrux'].keys():
                    reward += DESTROY_HORCRUX_REWARD

        return reward

    def apply_action(self, state, action):
        if action == "reset":
            return [copy.deepcopy(self.start_state)], [1.0]

        if action == "terminate":
            return [copy.deepcopy(state)], [1.0]

        next_state = copy.deepcopy(state)

        # Apply wizard actions
        for act in action:
            if act[0] == 'move':
                next_state['wizards'][act[1]] = act[2]
            elif act[0] == 'destroy' and len(h_name := [h for h, loc in state['horcrux'].items() if loc == act[2]]) > 0:
                horcrux_name = h_name[0]
                next_state['horcrux'] = {hor: state['horcrux'][hor] for hor in state['horcrux'].keys() if
                                         hor != horcrux_name}

        # Generate death eater transitions
        de_states = []
        de_probs = []

        for de in state['death_eaters']:
            path = self.initial['death_eaters'][de]['path']
            curr_loc = state['death_eaters'][de]

            # Find current position in path
            curr_pos = [i for i, loc in enumerate(path) if loc == curr_loc]
            if not curr_pos:  # If location not found in path (shouldn't happen), just stay
                new_state = copy.deepcopy(next_state)
                de_states.append(new_state)
                de_probs.append(1.0)
                continue

            curr_idx = curr_pos[0]
            # At start of path
            if curr_idx == 0:
                if len(path) > 1:
                    possible_moves = [(path[0], 0.5), (path[1], 0.5)]  # Stay
                else:
                    possible_moves = [(path[0], 1)]

            # At end of path
            elif curr_idx == len(path) - 1:
                possible_moves = [(path[curr_idx], 0.5), (path[curr_idx - 1], 0.5)]

            # Middle of path
            else:
                possible_moves = [
                    (path[curr_idx - 1], 1 / 3),  # Back
                    (path[curr_idx], 1 / 3),  # Stay
                    (path[curr_idx + 1], 1 / 3)  # Forward
                ]

            for new_loc, prob in possible_moves:
                new_state = copy.deepcopy(next_state)
                new_state['death_eaters'][de] = new_loc
                de_states.append(new_state)
                de_probs.append(prob)

        final_states = []
        final_probs = []

        for de_state, de_prob in zip(de_states, de_probs):
            for horcrux in state['horcrux'].keys():
                if horcrux in de_state['horcrux']:
                    prob_change = self.initial['horcrux'][horcrux]['prob_change_location']

                    # Stay in current location
                    stay_state = copy.deepcopy(de_state)
                    final_states.append(stay_state)
                    final_probs.append(de_prob * (1 - prob_change))

                    # Move to possible locations
                    possible_locs = self.initial['horcrux'][horcrux]['possible_locations']
                    prob_per_loc = prob_change / len(possible_locs)

                    for loc in possible_locs:
                        new_state = copy.deepcopy(de_state)
                        new_state['horcrux'][horcrux] = loc
                        final_states.append(new_state)
                        final_probs.append(de_prob * prob_per_loc)

        return final_states, final_probs

    def Value_Iteration(self):
        cached_values = self.cache.load_cached_values(self.initial, self.rounds)
        if cached_values is not None:
            return json.dumps(cached_values)
        V = []
        for t in range(self.rounds + 1):
            vs = {}
            for state in self.all_states:
                state_key = str(self.state_to_key(state))
                actions = self.get_actions(state)
                action_values = []

                for action in actions:
                    immediate_reward = self.calculate_reward(state, action)
                    new_states, probs = self.apply_action(state, action)

                    assert abs(sum(probs) - 1.0) < 1e-10  # Verify probabilities sum to 1

                    GAMMA = 0.9  # Add this at the top of the class

                    future_value = 0
                    if t > 0:
                        if action == "terminate":
                            future_value = 0
                        elif action == "reset":
                            initial_key = str(self.state_to_key(self.start_state))
                            future_value = GAMMA * V[t - 1][initial_key]['score']  # Apply gamma here
                        else:
                            for n_state, prob in zip(new_states, probs):
                                next_key = str(self.state_to_key(n_state))
                                future_value += prob * GAMMA * V[t - 1][next_key]['score']  # Apply gamma here

                    total_value = immediate_reward + future_value
                    action_values.append((action, total_value))

                best_action, max_score = max(action_values, key=lambda x: x[1])
                vs[state_key] = {'action': best_action, 'score': max_score}
            V.append(vs)
        self.cache.save_values(self.initial, self.rounds, V)
        return json.dumps(V)

    def convert_to_tuples(self, data):
        if isinstance(data, list):  # If the current element is a list
            return tuple(self.convert_to_tuples(item) for item in data)  # Convert to tuple and recurse
        elif isinstance(data, dict):  # If it's a dictionary
            return {key: self.convert_to_tuples(value) for key, value in data.items()}  # Recurse on dict values
        else:
            return data  # Leave other elements (like strings, numbers) unchanged

    def act(self, state):
        values = json.loads(self.V)
        state_key = str(self.state_to_key(self.simplify_state(state)))
        best_action = values[max(0, self.time)][state_key]['action']
        self.time -= 1
        return self.convert_to_tuples(best_action)


class WizardAgent:
    def __init__(self, initial):
        super().__init__(initial)
        raise NotImplementedError


    def act(self, state):
        raise NotImplementedError