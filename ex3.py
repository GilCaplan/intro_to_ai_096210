import itertools
from operator import matmul
from itertools import product
from collections import defaultdict
from utils import matrix_multiplication
ids = ["111111111", "222222222"]

DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1

class OptimalWizardAgent:
    def __init__(self, initial):
        self.state = initial
        self.map = initial['map']
        self.wizards = initial['wizards']
        self.horcrux = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.hor_pi0 = {key: [1 if v == self.horcrux[key]['location'] else 0 for v in self.horcrux[key]['possible_locations']] for key in self.horcrux.keys()}
        self.hor_transition = self.calculate_transition_matrix(True)

        self.death_eaters_pi0 = {key: [1 if c == self.death_eaters[key]['index'] else 0 for c, _ in enumerate(self.death_eaters[key]['path'])]for key in self.death_eaters.keys()}
        self.death_eaters_transition = self.calculate_transition_matrix(False)
        self.num_horcruxes = len(self.horcrux)
        self.flatten_hor = []
        self.flatten_death = []
        for key in self.death_eaters.keys():
            self.flatten_death.extend(self.death_eaters[key]['path'])
        for key in self.horcrux.keys():
            self.flatten_hor.extend(self.horcrux[key]['possible_locations'])
        self.all_states = self.compute_states()
        self.R = self.calculate_reward(True)
        self.rounds = initial['turns_to_go']
        self.lr = 0.01
        self.V = self.Value_Iteration()
        self.time = 0

    def compute_states(self):
        states = []
        wiz_locs, death_eaters_locs, horcrux_locs = {}, {}, {}
        for wizards in self.wizards:
            local_wiz = []
            for i in range(len(self.map)):
                for j in range(len(self.map[i])):
                    if self.map[i][j] == 'P':
                        local_wiz.append((i, j))
            wiz_locs[wizards] = local_wiz

        for de in self.death_eaters:
            death_eaters_locs[de] = [(i, j) for i, j in self.death_eaters[de]['path']]

        for hor in self.horcrux:
            horcrux_locs[hor] = [(i, j) for i, j in self.horcrux[hor]['possible_locations']] + [None]

        wizard_combinations = product(*[wiz_locs[wizard] for wizard in sorted(self.wizards.keys())])
        de_combinations = product(*[death_eaters_locs[de] for de in sorted(self.death_eaters.keys())])
        horcrux_combinations = product(*[horcrux_locs[hor] for hor in sorted(self.horcrux.keys())])

        for wiz_pos in wizard_combinations:
            for de_pos in de_combinations:
                for hor_pos in horcrux_combinations:
                    state = {'wizards': {}, 'death_eaters': {}, 'horcrux': {}}
                    for i, wizard in enumerate(sorted(self.wizards.keys())):
                        state['wizards'][wizard] = {'location': wiz_pos[i]}
                    for i, de in enumerate(sorted(self.death_eaters.keys())):
                        state['death_eaters'][de] = {'index': de_pos[i],
                                'path': self.death_eaters[de]['path']}

                    for i, hor in enumerate(sorted(self.horcrux.keys())):
                        state['horcrux'][hor] = {
                                'location': hor_pos[i],
                                'possible_locations': self.horcrux[hor]['possible_locations'],
                                'prob_change_location': self.horcrux[hor]['prob_change_location']
                        }
                        states.append(state)
        return states

    def get_actions(self, state):
        actions = []
        horcruxes = state['horcrux']
        wizards = state['wizards']
        def get_move_actions(loc, wiz_name):
            move_actions = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                if 0 <= new_loc[0] < len(self.map) and 0 <= new_loc[1] < len(self.map[0]):
                    if self.map[new_loc[0]][new_loc[1]] != 'I':
                        move_actions.append(('move', wiz_name, new_loc))
            return tuple(move_actions)

        def get_destroy_horcrux_actions(loc, wiz_name):
            destroy_actions = []
            if (loc[0], loc[1]) in [horcruxes[hor]['location'] for hor in horcruxes]:
                destroy_actions.append(('destroy', wiz_name, (loc[0], loc[1])))
            return tuple(destroy_actions)

        def get_wait_actions(wiz_name):
            return tuple([('wait', wiz_name)])

        for wizard in wizards:
            wiz_loc = wizards[wizard]['location']
            actions.extend(get_wait_actions(wiz_loc) + get_move_actions(wiz_loc, wizard) + get_destroy_horcrux_actions(wiz_loc, wizard))

        return tuple(itertools.product(*actions)) + tuple([('termination',), ('reset',)])

    def state_to_key(self, state):
        """Convert state dict to a hashable tuple for dictionary keys"""
        # Convert wizards locations to tuples
        wizard_locs = tuple(sorted((name, tuple(info['location']))
                                   for name, info in state['wizards'].items()))

        # Convert death eaters indices to tuples
        de_positions = tuple(sorted((name, info['index'])
                                    for name, info in state['death_eaters'].items()))

        # Convert horcrux locations to tuples (handling None case)
        horcrux_locs = tuple(sorted((name, tuple(info['location']) if info['location'] is not None else None)
                                    for name, info in state['horcrux'].items()))

        return (wizard_locs, de_positions, horcrux_locs)

    def is_wizard_at_location(self, state, location):
        """Check if any wizard is at the given location"""
        return any(wizard['location'] == location
                   for wizard in state['wizards'].values())
    def can_wizard_destroy(self, state, horcrux_location):
        """Check if any wizard can destroy a horcrux at given location"""
        if horcrux_location is None:
            return False
        return any(wizard['location'] == horcrux_location
                   for wizard in state['wizards'].values())

    def Value_Iteration(self):
        # Initialize V[0] with immediate rewards for each state
        V = [{self.state_to_key(state): {'values': [(None, self.calculate_reward(True))]}
              for state in self.all_states}]

        # Iterate for t rounds
        for t in range(1, self.rounds):
            vs = {}
            for state in self.all_states:
                actions = self.get_actions(state)
                action_values = []

                for action in actions:
                    # Immediate reward for this state
                    immediate_reward = self.calculate_reward(True)

                    # Future expected reward based on transitions
                    future_reward = 0

                    # Death eater probability contribution
                    for de in state['death_eaters']:
                        next_de_probs = matrix_multiplication([self.death_eaters_pi0[de]], self.death_eaters_transition[de])[0]  # Extract row
                        future_reward += DEATH_EATER_CATCH_REWARD * sum(prob for prob in next_de_probs)
                    # Horcrux probability contribution
                    for h in state['horcrux']:
                        next_hor_probs = matrix_multiplication([self.hor_pi0[h]], self.hor_transition[h])[0]
                        future_reward += DESTROY_HORCRUX_REWARD * sum(prob for prob in next_hor_probs)

                    # V[t] = R(s) + Σ P(s'|s,a) * V[t-1](s')
                    total_value = immediate_reward + future_reward * V[t - 1][self.state_to_key(state)]['values'][0][
                        1]  # Using best value from previous iteration
                    action_values.append((action, total_value))

                # Store all action-value pairs for this state
                vs[self.state_to_key(state)] = {'values': action_values}

            V.append(vs)

        return V
    def calculate_reward(self, simple=False):
        reward = 0
        if simple:
            for wizard in self.wizards.keys():
                wx, wy = self.wizards[wizard]["location"]
                if (wx, wy) in self.flatten_death:
                    reward += DEATH_EATER_CATCH_REWARD
                if (wx, wy) in self.flatten_hor:
                    reward += DESTROY_HORCRUX_REWARD
            return reward
        for wizard in self.wizards.keys():
            wx, wy = wizard["location"]
            # if we're on a horcrux then 2 * probability that we can destroy it
            # if a wizard encounters a death eater -1 * that probability
            reward += sum(2 * matmul(self.hor_pi0[wx], hor[:, wy]) for hor in self.hor_transition.values())
            reward -= sum(matmul(self.death_eaters_pi0[wx], de[:, wy]) for de in self.death_eaters_transition.values())
        return reward
    def calculate_transition_matrix(self, horcrux_transition=False):
        if horcrux_transition:
            m = {}
            # for each horcrux calculating transition matrix
            # 1 - p on diagnol, rest is choosing uniform => p / length
            for key in self.horcrux.keys():
                locs_num = len(self.horcrux[key]['possible_locations'])
                p = self.horcrux[key]['prob_change_location']
                t = []
                for i in range(locs_num):
                    t.append([1 - p if i == j else p / locs_num  for j in range(locs_num)])
                m[key] = t
            return m
        # for deatheaters movement
        m = {}
        def p(i, j, key):
            return 1 / (sum(1 if (i + xi, j + yi) in self.death_eaters[key]['path'] else 0
                            for xi, yi in [(-1, 0), (1, 0), (0, -1), (0, 1)]) + 1)

        for key in self.death_eaters.keys():
            n = len(self.death_eaters[key]['path'])
            m[key] = [[p(i, j, key) for j in range(n)] for i in range(n)]
        return m

    def act(self, state):
        state_key = self.state_to_key(state)
        values = self.V[self.time][state_key]  # Get all action-value pairs
        self.time +=1
        best_action = max(values, key=lambda x: x[1])[0]  # Choose action with highest value
        return best_action

class WizardAgent:
    def __init__(self, initial):
        super().__init__(initial)
        raise NotImplementedError


    def act(self, state):
        raise NotImplementedError
