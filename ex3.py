from collections import defaultdict
from utils import PriorityQueue
from itertools import product

class WizardAgent:

    def extract_character_data(self, character_dict):
        character_queue = PriorityQueue(min, lambda x: x[0])
        for character_name, character_data in character_dict.items():
            character_queue.append((character_name, character_data))
        return character_queue

    def extract_initial_info(self, state_dict):
        horcrux_dict = state_dict['horcrux']
        death_eater_dict = state_dict['death_eaters']

        horcrux_queue = self.extract_character_data(horcrux_dict)
        death_eater_queue = self.extract_character_data(death_eater_dict)

        horcrux_names = death_eater_names = horcrux_locs = horcrux_probs = \
        death_eater_indices = death_eater_locs = list()

        while horcrux_queue:
            horcrux_name, horcrux_data = horcrux_queue.pop()
            horcrux_names.append(horcrux_name)
            horcrux_locs.append(horcrux_data['possible_locations'])
            horcrux_probs.append(horcrux_data['prob_change_location'])

        while death_eater_queue:
            death_eater_name, death_eater_data = death_eater_queue.pop()
            death_eater_names.append(death_eater_name)
            death_eater_indices.append(list(range(len(death_eater_data['path']))))
            death_eater_locs.append(death_eater_data['path'])

        return horcrux_names, horcrux_locs, horcrux_probs, death_eater_names, death_eater_indices, death_eater_locs
    
    def generate_all_environments(self):
        horcrux_combinations = list(product(*self.horcrux_locs))
        death_eater_combinations = list(product(*self.death_eater_indices))

        wizard_possible_locs = [[] for _ in self.wizards_names]
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i,j] == 'P':
                    [wizard_possible_locs[i].append((i, j))]

        wiz_combinations = list(product(*wizard_possible_locs))
        all_environments = list(product(horcrux_combinations, death_eater_combinations))
        all_states = list(product(wiz_combinations, horcrux_combinations, death_eater_combinations))

        return all_environments, all_states
    
    def get_next_locs_probs_horcrux(self, hor_loc, hor_locs, hor_prob_change):
        next_locs = dict()
        next_locs[hor_loc] = (1 - hor_prob_change)
        distr = hor_prob_change / len(hor_locs)
        for loc in hor_locs:
            if loc == hor_loc:
                next_locs[hor_loc] += distr
            else:
                next_locs[loc] = distr
        return next_locs
    
    def get_next_locs_probs_death_eaters(self, de_index, de_indices):
        next_locs = dict()
        if de_index == 0:
            next_locs[de_index] = 0.5
            next_locs[(de_index + 1)] = 0.5
        elif de_index == (len(de_indices) - 1):
            next_locs[de_index] = 0.5
            next_locs[(de_index - 1)] = 0.5
        else:
            next_locs[de_index] = (1 / 3)
            next_locs[(de_index - 1)] = (1 / 3)
            next_locs[(de_index + 1)] = (1 / 3)
        return next_locs


    def get_transition_probabilities(self):    
        # combo AKA environment -> (horcruxes_tuple, deatheater_tuple)
        next_combo_dict = dict()
        for combo in self.all_environments:
            next_state_distribution = defaultdict()
            next_hor_from_hor_locs = list()
            next_de_ind_from_ind = list()
            horcrux_locs_env = combo[0]
            deatheater_indices = combo[1]
            for i, horcrux_loc in enumerate(horcrux_locs_env):
                hor_prob = self.horcrux_probs[i]
                hor_possible_locs = self.horcrux_locs[i]
                next_hor_locs = self.get_next_locs_probs_horcrux(horcrux_loc, hor_possible_locs, hor_prob)
                next_hor_from_hor_locs.append(next_hor_locs)
            for j, de_index in enumerate(deatheater_indices):
                de_possible_indices = self.death_eater_indices[j]
                next_de_indices = self.get_next_locs_probs_death_eaters(de_index, de_possible_indices)
                next_de_ind_from_ind.append(next_de_indices)

            for hor_states in product(*[list(hor.keys()) for hor in next_hor_from_hor_locs]):
                for de_states in product(*[list(de.keys()) for de in next_de_ind_from_ind]):
                    prob = 1
                    for i, hor_state in enumerate(hor_states):
                        prob *= next_hor_from_hor_locs[i][hor_state]
                    for j, de_state in enumerate(de_states):
                        prob *= next_de_ind_from_ind[j][de_state]
                    
                    next_state_distribution[(tuple(hor_states), tuple(de_states))] += prob

            next_combo_dict[combo] = dict(next_state_distribution)

        return next_combo_dict

    def get_all_actions(self, state):
        wiz_locs = state[0]
        hor_locs = state[1]
        de_locs = state[2]

        def get_move_actions(wiz_locs, wiz_names):
            offsets = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            move_actions = [[] for _ in wiz_locs]
            for i, (loc, name) in enumerate(zip(wiz_locs, wiz_names)):
                wiz_actions = []
                for offset in offsets:
                    new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                    if 0 <= new_loc[0] < len(self.map) and \
                       0 <= new_loc[1] < len(self.map[0]) and\
                       self.map[new_loc[0], new_loc[1]] == 'P':
                        wiz_actions.append(('move', name, new_loc))
                move_actions[i].extend(wiz_actions)
            return move_actions
        
        def get_destroy_actions(wiz_locs, horcrux_locs, wiz_names, horcrux_names):
            destroy_actions = [[] for _ in wiz_locs]
            for i, (loc, name) in enumerate(zip(wiz_locs, wiz_names)):
                wiz_actions = []
                for hor_loc, hor_name in zip(horcrux_locs, horcrux_names):
                    if loc == hor_loc:
                        wiz_actions.append(('destroy', name, hor_name))
                destroy_actions[i].extend(wiz_actions)
            return destroy_actions

        def get_wait_actions(wiz_names):
            return [[('wait', wiz_name)] for wiz_name in wiz_names]
        
        actions = [[] for _ in wiz_locs]
        move_actions = get_move_actions(wiz_locs, self.wizards_names)
        destroy_actions = get_destroy_actions(wiz_locs, hor_locs, self.wizards_names, self.horcrux_names)
        wait_actions = get_wait_actions(self.wizards_names)
        [actions[i].extend(move_actions[i]) for i in range(len(wiz_locs))]
        [actions[i].extend(destroy_actions[i]) for i in range(len(wiz_locs))] 
        [actions[i].extend(wait_actions[i]) for i in range(wiz_locs)]  

        actions = list(product(*actions))
        actions.extend(["reset", "terminate"]) 

        return actions
    
    def state_to_list(self, state):
        wizards_queue = self.extract_character_data(state['wizards'])
        horcrux_queue = self.extract_character_data(state['horcrux'])
        de_queue = self.extract_character_data(state['death_eaters'])

        wizards_locs = []
        while wizards_queue:
            wizard_name, wizard_data = wizards_queue.pop()
            wizards_locs.append(wizard_data['location'])
        wizards_locs = tuple(wizards_locs)
        horcrux_locs = []
        while horcrux_queue:
            horcrux_name, horcrux_data = horcrux_queue.pop()
            horcrux_locs.append(horcrux_data['location'])
        horcrux_locs = tuple(horcrux_locs)
        de_inds = []
        while de_queue:
            de_name, de_data = de_queue.pop()
            de_inds.append(de_data['index'])
        de_inds = tuple(de_inds)
        return (wizards_locs, horcrux_locs, de_inds)
    

    def value_iteration(self):
        policy = [{} for _ in range(1, self.turns + 1)]
        for state in self.all_states:
            pass

    def __init__(self, initial):
        self.horcrux_names, self.horcrux_locs, self.horcrux_probs, self.death_eater_names,\
              self.death_eater_indices, self.death_eater_locs = self.extract_initial_info(initial)
        self.all_environments, self.all_states = self.generate_all_environments()
        self.stochastic_environments_probs = self.get_transition_probabilities()
        self.map = initial['map']
        self.wizards_names = list(initial['wizards'].keys())
        self.turns = initial['turns_to_go']
        self.policy = self.value_iteration()

    def act(self, state):
        state_list = self.state_to_list(state)
        turns_to_go = state['turns_to_go']
        actions = self.policy[turns_to_go][state_list]
        
class OptimalWizardAgent(WizardAgent):
    def __init__(self, initial):
        pass

    def act(self, state):
        pass