import copy
import itertools
import time
from copy import deepcopy
from itertools import product
import json
import os
import hashlib
ids = ["111111111", "222222222"]

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1


def moves_towards_horcrux(state, action1, action2):
    """Returns True if action1 moves wizards closer to horcruxes than action2."""

    def total_distance_to_horcruxes(state, action):
        """Computes sum of wizard distances to nearest horcrux after taking an action."""
        next_state = apply_action(state, action)
        horcrux_locs = list(hor for hor in state['horcrux'].values() if hor is not None)
        if not horcrux_locs:  # No horcruxes left
            return float('-inf')

        total_distance = 0
        for wizard, wiz_loc in next_state['wizards'].items():
            min_dist = min(manhattan_distance(wiz_loc, h) for h in horcrux_locs)
            total_distance += min_dist
        return total_distance

    return total_distance_to_horcruxes(state, action1) < total_distance_to_horcruxes(state, action2)


def manhattan_distance(p1, p2):
    """Computes Manhattan distance between two points (row1, col1) and (row2, col2)."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def state_to_key(state):
    wiz = tuple((name, loc) for name, loc in state['wizards'].items())
    de = tuple((name, idx) for name, idx in state['death_eaters'].items())
    hor = tuple((name, loc) for name, loc in state['horcrux'].items() if loc is not None)
    return str(wiz + de + hor)

def apply_action(state, action):
    next_state = copy.deepcopy(state)
    if isinstance(action, tuple):
        for act in action:
            if act[0] == 'move':
                next_state['wizards'][act[1]] = act[2]
    return next_state


def calculate_reward(initial, state, action):
    reward = 0
    if action == "reset":
        return RESET_REWARD
    if isinstance(action, tuple):
        de_locations = set((name, initial["death_eaters"][name]["path"][idx]) for name, idx in state['death_eaters'].items())
        for loc in [wiz_loc for _, wiz_loc in state['wizards'].items()]:
            if loc in de_locations:
                reward += DEATH_EATER_CATCH_REWARD
        hor_dest = []
        for act in action:
            if act[0] == 'destroy' and act[2] not in hor_dest:
                if act[2] in state['horcrux'].keys() and state['wizards'][act[1]] == state['horcrux'][act[2]]:
                    reward += DESTROY_HORCRUX_REWARD
                    hor_dest.append(act[2])
    return reward


def simplify_state(state):
    return {
        'death_eaters': {key: info['index'] for key, info in state['death_eaters'].items()},
        'horcrux': {key: value['location'] for key, value in state['horcrux'].items()},
        'wizards': {key: value['location'] for key, value in state['wizards'].items()}}


class OptimalWizardAgent:
    def __init__(self, initial):
        self.start_state = simplify_state(initial)
        self.initial = copy.deepcopy(initial)
        self.map = initial['map']
        self.wizards = initial['wizards']
        self.horcrux = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.all_states = self.compute_states()
        self.rounds = initial['turns_to_go']
        self.cache = ValueIterationCache()
        self.V = self.Value_Iteration()
        self.time = initial['turns_to_go']

    def compute_states(self):
        wiz_locs, death_eaters_locs, horcrux_locs = {}, {}, {}
        for wizard in self.wizards:
            local_wiz = []
            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    if self.map[i][j] == 'P':
                        local_wiz.append((i, j))
            wiz_locs[wizard] = local_wiz

        for de in self.death_eaters:
            death_eaters_locs[de] = [i for i in range(len(self.death_eaters[de]['path']))]

        horcrux_locs = {}
        for hor in self.horcrux:
            horcrux_locs[hor] = [loc for loc in self.horcrux[hor]['possible_locations']] + [None]

        wizard_combinations = tuple(product(*[wiz_locs[wizard] for wizard in wiz_locs]))
        de_combinations = tuple(product(*[death_eaters_locs[de] for de in death_eaters_locs]))
        horcrux_combinations = tuple(product(*[horcrux_locs[hor] for hor in horcrux_locs]))

        states = []
        if self.start_state not in states:
            states.append(self.start_state)

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

                    if state_to_key(state) != state_to_key(self.start_state):
                        states.append(state)

        return states

    def get_actions(self, state):
        actions = []
        wizards = state['wizards']

        def get_move_actions(loc, wiz_name):
            move_actions = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                if 0 <= new_loc[0] < len(self.map) and 0 <= new_loc[1] < len(self.map[0]):
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

    def apply_action(self, state, action):
        if action in ["reset", "terminate"]:
            return [self.start_state if action == "reset" else state], [1.0]

        next_state = copy.deepcopy(state)
        for act in action:
            if act[0] == 'move':
                next_state['wizards'][act[1]] = act[2]
            elif act[0] == 'destroy' and act[2] in state['horcrux'] and state['wizards'][act[1]] == state['horcrux'][
                act[2]]:
                next_state['horcrux'].pop(act[2], None)

        de_states = [next_state]
        de_probs = [1.0]

        for de_name in state['death_eaters'].keys():
            path = self.initial['death_eaters'][de_name]['path']
            curr_idx = state['death_eaters'][de_name]
            path_len = len(path)

            new_de_states = []
            new_de_probs = []

            for curr_state, curr_prob in zip(de_states, de_probs):
                if path_len == 1:
                    moves = [(curr_idx, 1.0)]
                elif curr_idx == 0:
                    moves = [(0, 0.5), (1, 0.5)]
                elif curr_idx == path_len - 1:
                    moves = [(curr_idx, 0.5), (curr_idx - 1, 0.5)]
                else:
                    moves = [(curr_idx - 1, 1 / 3), (curr_idx, 1 / 3), (curr_idx + 1, 1 / 3)]

                for new_idx, move_prob in moves:
                    new_state = copy.deepcopy(curr_state)
                    new_state['death_eaters'][de_name] = new_idx
                    new_de_states.append(new_state)
                    new_de_probs.append(curr_prob * move_prob)

            de_states = new_de_states
            de_probs = new_de_probs

        if not state['horcrux']:
            return de_states, de_probs

        final_states = []
        final_probs = []

        for de_state, de_prob in zip(de_states, de_probs):
            horcrux_states = [de_state]
            horcrux_probs = [1.0]

            for horcrux in list(state['horcrux'].keys()):
                if horcrux not in self.initial['horcrux']:
                    continue

                hor_info = self.initial['horcrux'][horcrux]
                p_change = hor_info['prob_change_location']
                locations = hor_info['possible_locations']
                num_locs = len(locations)

                new_states = []
                new_probs = []

                for curr_state, curr_prob in zip(horcrux_states, horcrux_probs):
                    if horcrux not in curr_state['horcrux']:
                        new_states.append(curr_state)
                        new_probs.append(curr_prob)
                        continue

                    curr_loc = curr_state['horcrux'][horcrux]
                    stay_prob = 1 - p_change + (p_change / num_locs if curr_loc in locations else 0)

                    stay_state = copy.deepcopy(curr_state)
                    new_states.append(stay_state)
                    new_probs.append(curr_prob * stay_prob)

                    move_prob = p_change / num_locs
                    for new_loc in locations:
                        if new_loc != curr_loc:
                            move_state = copy.deepcopy(curr_state)
                            move_state['horcrux'][horcrux] = new_loc
                            new_states.append(move_state)
                            new_probs.append(curr_prob * move_prob)

                horcrux_states = new_states
                horcrux_probs = new_probs

            final_states.extend(horcrux_states)
            final_probs.extend(p * de_prob for p in horcrux_probs)

        prob_sum = sum(final_probs)
        if prob_sum > 0:
            final_probs = [p / prob_sum for p in final_probs]

        return final_states, final_probs

    def Value_Iteration(self):
        cached_values = self.cache.load_cached_values(self.initial, self.rounds)
        if cached_values is not None:
            return json.dumps(cached_values)

        V = []
        for t in range(self.rounds + 1):
            vs = {}
            for state in self.all_states:
                state_key = state_to_key(state)
                if t == 0:
                    vs[state_key] = {'action': 'terminate', 'score': 0}
                    continue

                actions = self.get_actions(state)
                best_score = float('-inf')
                best_action = None

                for action in actions:
                    immediate_reward = calculate_reward(self.initial, state, action)
                    if action == "terminate":
                        total_value = immediate_reward
                    else:
                        new_states, probs = self.apply_action(state, action)
                        future_value = 0
                        for next_state, prob in zip(new_states, probs):
                            future_value += prob * V[t - 1][state_to_key(next_state)]['score']
                        total_value = immediate_reward + future_value

                    if total_value > best_score:
                        best_score = total_value
                        best_action = action
                    elif total_value == best_score:
                        if len(state['horcrux']) == 0 and "reset" in actions:
                            best_action = "reset"
                        elif moves_towards_horcrux(state, action, best_action):
                                best_action = action

                vs[state_key] = {'action': best_action, 'score': best_score}
            V.append(vs)

        self.cache.save_values(self.initial, self.rounds, V)
        return json.dumps(V)

    def convert_to_tuples(self, data):
        if isinstance(data, list):
            return tuple(self.convert_to_tuples(item) for item in data)
        elif isinstance(data, dict):
            return {key: self.convert_to_tuples(value) for key, value in data.items()}
        else:
            return data

    def act(self, state):
        values = json.loads(self.V)
        state_key = str(state_to_key(simplify_state(state)))
        best_action = values[max(0, self.time)][state_key]['action']
        self.time -= 1
        return self.convert_to_tuples(best_action)


class WizardAgent(OptimalWizardAgent):
    def __init__(self, initial):
        super().__init__(initial)

    def act(self, state):
        values = json.loads(self.V)
        state_key = str(state_to_key(simplify_state(state)))
        best_action = values[max(0, self.time)][state_key]['action']
        self.time -= 1
        return self.convert_to_tuples(best_action)

def clear_cache():
    """Delete all cached Value Iteration files."""
    if os.path.exists("vi_cache"):
        for file in os.listdir("vi_cache"):
            file_path = os.path.join("vi_cache", file)
            try:
                os.remove(file_path)
            except OSError:
                pass

class ValueIterationCache:
    def __init__(self, cache_dir="vi_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _generate_state_hash(self, initial_state):
        """Generate a unique hash for the initial state configuration"""
        # Convert the initial state to a stable string representation
        state_str = json.dumps(initial_state, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def _get_cache_path(self, state_hash, rounds):
        """Get the cache file path for a specific state and number of rounds"""
        return os.path.join(self.cache_dir, f"vi_{state_hash}_{rounds}.json")

    def load_cached_values(self, initial_state, rounds):
        """Load cached Value Iteration results if they exist"""
        state_hash = self._generate_state_hash(initial_state)
        cache_path = self._get_cache_path(state_hash, rounds)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def save_values(self, initial_state, rounds, values):
        """Save Value Iteration results to cache"""
        state_hash = self._generate_state_hash(initial_state)
        cache_path = self._get_cache_path(state_hash, rounds)

        try:
            with open(cache_path, 'w') as f:
                json.dump(values, f)
            return True
        except IOError:
            return False

