from itertools import product
from copy import deepcopy
from time import time

ids = ["342663978", "207840240"]

def get_base_loc(map):
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] == 'B':
                return (i, j)
    return None

def is_adjacent(loc1, loc2):
    return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1]) == 1



class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = deepcopy(initial)
        self.map = initial['map']
        self.base = get_base_loc(self.map)
        self.turns = self.initial['turns to go']
        self.pirate_names = list(self.initial['pirate_ships'].keys())
        self.prob_change_location = {treasure: self.initial['treasures'][treasure]['prob_change_location'] for treasure in self.initial['treasures']}
        self.initial_state = ((self.base, 2), \
                                    tuple([tuple([(name, self.initial['marine_ships'][name]['index']) for name in self.initial['marine_ships']]), \
                                     tuple([(name, self.initial['treasures'][name]['location']) for name in self.initial['treasures']])]))

        self.marine_possible_indexes = self.get_possible_movements(self.initial['marine_ships'], 'path')
        self.treasures_possible_locations = self.get_possible_movements(self.initial['treasures'], 'possible_locations')

        self.pirate_possible_locations = [(i, j) for i in range(len(self.map)) for j in range(len(self.map[0]))\
                                           if self.map[i][j] == 'S' or self.map[i][j] == 'B']
        
        self.stochastic_enviroments = self.get_all_enviroments()
        self.states = self.get_all_states()
        self.actions_from_state = self.get_actions_from_state(self.states)
        self.transition_probabilities = self.get_transition_probabilities(self.stochastic_enviroments)
        self.full_transition_probabilities = self.full_transition_probabilities()
        self.policy = self.make_policy()
        x = 0
        
    
    def make_policy(self):
        policy = list()
        for i in range(self.turns + 1):
                policy.append(dict())
        for state in self.states:
            
            pirate_loc, capacity = state[0]
            marines, treasures = state[1]
            if pirate_loc in [self.initial['marine_ships'][name]['path'][idx] for (name, idx) in marines]:
                policy[0][state] = (None, -len(self.pirate_names))
            else:
                policy[0][state] = (None, 0)
            
        for turn in range(1, self.turns + 1):
            for state in self.states:
                deterministic_state = state[0]
                pirate_loc, capacity = deterministic_state
                stochastic_state = state[1]
                marines, treasures = stochastic_state
                if self.collision(pirate_loc, marines):
                    instant_reward = -len(self.pirate_names)
                else:
                    instant_reward = 0

                actions = self.actions_from_state[state]
                best_action = 'terminate'
                best_value = 0
                for action in actions:
                    action_reward = self.action_reward(action, state)
                    expected_value = 0
                    for next_state, probability in self.full_transition_probabilities[(state, action)].items():
                        expected_value += (probability * (policy[turn - 1][next_state][1]))
                    if action_reward + expected_value > best_value:
                        best_value = action_reward + expected_value
                        best_action = action
                policy[turn][state] = (best_action, best_value + instant_reward)
        return policy


    def collision(self, new_pirate_loc, marines):
            for marine_name, marine_index in marines:
                if new_pirate_loc == self.initial['marine_ships'][marine_name]['path'][marine_index]:
                    return True
            return False


    def action_reward(self, action, state):
        capacity = state[0][1]
        if action == 'reset':
            return -2
        if action == 'terminate':
            return 0
        if action[0][0] == 'collect':
            return 0
        if action[0][0] == 'sail':
            return 0
        if action[0][0] == 'deposit':
            return len(self.pirate_names) * (2 - capacity) * 4
        if action[0][0] == 'wait':
            return 0
        

    def full_transition_probabilities(self):
        transitions = dict()
        for state in self.states:
            for action in self.actions_from_state[state]:
                transitions[(state, action)] = self.states_from_action(state, action)

        return transitions

    def states_from_action(self, state, action):

        states_probs = {}

        deterministic_state = state[0]
        pirate_loc, capacity = deterministic_state
        stochastic_state = state[1]

        
        for next_env, probability in self.transition_probabilities[stochastic_state].items():
            if action == 'terminate':
                return {}

            if action == 'reset':
                return {self.initial_state: 1}

            if action[0][0] == 'collect':
                new_capacity = capacity - 1
                if self.collision(pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'sail':
                new_pirate_loc = action[0][2]
                new_capacity = capacity
                if self.collision(new_pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((new_pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'deposit':
                new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'wait':
                new_capacity = capacity
                if self.collision(pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability   
        return states_probs         
        


    def get_actions_from_state(self, states):
        """
        Get all possible actions from a given state

        :param states: a tuple of all possible states

        :return: a dictionary containing all possible actions from a given state
                    actions: (state: tuple, action: str)
        """

        actions_from_state = dict()

        for state in states:
            deterministic_state = state[0]
            pirate_loc, capacity = deterministic_state

            stochastic_state = state[1]
            marines, treasures = stochastic_state

            # Get all possible actions

            # terminate and reset actions
            actions = ['reset', 'terminate']

            # collect actions
            atomic = []
            for treasure_info in treasures:
                treasure_loc = treasure_info[1]
                treasure_name = treasure_info[0]

                if is_adjacent(pirate_loc, treasure_loc) and capacity > 0:
                    for pirate in self.pirate_names:
                        atomic.append(('collect', pirate, treasure_name))
                    actions.append(tuple(atomic))

            # sail actions
            
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (pirate_loc[0] + offset[0], pirate_loc[1] + offset[1]) in self.pirate_possible_locations:
                    atomic = []
                    for pirate in self.pirate_names:
                        atomic.append(('sail', pirate, (pirate_loc[0] + offset[0], pirate_loc[1] + offset[1])))
                    actions.append(tuple(atomic))

            # deposit actions
            atomic = []
            if pirate_loc == self.base and capacity < 2:
                for pirate in self.pirate_names:
                    atomic.append(('deposit', pirate))
                actions.append(tuple(atomic))

            # wait actions
            atomic = []
            for pirate in self.pirate_names:
                atomic.append(('wait', pirate))
            actions.append(tuple(atomic))

            # Generate all possible actions from the state
            actions_from_state[state] = actions

        return actions_from_state


    def get_transition_probabilities(self, stochastic_enviroments):
        
        transition = dict()
        for environment in stochastic_enviroments:
            transition[environment] = self.get_transition_probabilities_from_environment(environment)
        return transition


    def get_transition_probabilities_from_environment(self, stochastic_enviroment):

        marines, treasures = stochastic_enviroment

        marine_probs = dict()

        for marine_name, marine_index in marines:
            marine_path = self.initial['marine_ships'][marine_name]['path']

            if marine_index == len(marine_path) - 1:
                marine_next_locations_probs = {(marine_name, marine_index): 0.5, (marine_name, marine_index - 1): 0.5}
            elif marine_index == 0:
                marine_next_locations_probs = {(marine_name, marine_index): 0.5, (marine_name, marine_index + 1): 0.5}
            else:
                marine_next_locations_probs = {(marine_name, marine_index): 1/3, (marine_name, marine_index + 1): 1/3,\
                                                (marine_name, marine_index - 1): 1/3}
                
            
            marine_probs[marine_name] = marine_next_locations_probs

        treasure_probs = dict()
        for treasure_name, treasure_loc in treasures:
            treasure_next_locations_probs = dict()
            for possible_loc in self.initial['treasures'][treasure_name]['possible_locations']:
                if possible_loc == treasure_loc:
                    treasure_next_locations_probs[(treasure_name, possible_loc)] = 1 - self.prob_change_location[treasure_name]\
                                                                    + self.prob_change_location[treasure_name] / (len(self.initial['treasures'][treasure_name]['possible_locations']))  
                else:
                    treasure_next_locations_probs[(treasure_name, possible_loc)] = self.prob_change_location[treasure_name] / (len(self.initial['treasures'][treasure_name]['possible_locations']))
            
            treasure_probs[treasure_name] = treasure_next_locations_probs

        all_probs = []

        # Append marine transition probabilities
        for marine_name, transitions in marine_probs.items():
            all_probs.append([(marine_name, loc, prob) for ((marine_name, loc), prob) in transitions.items()])

        # Append treasure transition probabilities
        for treasure_name, transitions in treasure_probs.items():
            all_probs.append([(treasure_name, loc, prob) for ((treasure_name, loc), prob) in transitions.items()])

        # Dictionary to hold full transition probabilities
        full_transition_probabilities = {}

        # Compute all possible combinations of states using Cartesian product
        for combined_states in product(*all_probs):
            env = [[], []]
            p = 1
            for (name, loc, prob) in combined_states:
                p *= prob
                if type(loc) == int:
                    env[0].append((name, loc))
                else:
                    env[1].append((name, loc))
            env[0] = tuple(env[0])
            env[1] = tuple(env[1])
            full_transition_probabilities[tuple(env)] = p

        return full_transition_probabilities



    def get_all_states(self):
        """
        Get all possible states for the game

        :return: a tuple of all possible states
                    state: tuple(deterministic_state: tuple, stochastic_state: tuple)
        """

        deterministic_enviroments = tuple([(pirate_loc, capacity) for pirate_loc in self.pirate_possible_locations\
                                          for capacity in [0, 1, 2]])
        
        # Generate all combinations of deterministic_state and stochastic_state
        states = tuple(list(product(deterministic_enviroments, self.stochastic_enviroments)))
        return states


    def get_all_enviroments(self):
        """
        Get all possible enviroments for the game

        :return: a list of all possible enviroments
                    Environment: (marines_locations: tuple, treasures_locations: tuple)
                    entity_location: tuple(entity_name: str, location: tuple)
        """
        

        # Generate all combinations of marine_locations and treasure_locations
        all_marine_enviroments = list(product(*self.marine_possible_indexes))
        all_treasure_enviroments = list(product(*self.treasures_possible_locations))

        # Generate all possible enviroments
        enviroments = tuple(list(product(all_marine_enviroments, all_treasure_enviroments)))
        return enviroments


    def get_possible_movements(self, entity_info, criteria):
        """
        Get all possible movements for a given entity

        :param enity_info: a dictionary containing information about the entity
        :param criteria: the criteria to get the possible movements, either 'possible_locations' or 'path'

        :return: a tuple containing all possible entity name and the movements combinations
                    tuple(entity_name: str, possible_movements: tuple)
        """

        entity_possible_movements = list()
        for entity in entity_info:
            
            if criteria == 'possible_locations':
                entity_possible_movements.append(tuple([(entity, loc) for loc in entity_info[entity][criteria]]))
            else:
                entity_possible_movements.append(tuple([(entity, index) for index in range(len(entity_info[entity][criteria]))]))
            

        return entity_possible_movements
     


    def act(self, state):
        # make a tuple state of the state
        ship_name = list(state['pirate_ships'].keys())[0]
        pirate_loc = state['pirate_ships'][ship_name]['location']
        capacity = state['pirate_ships'][ship_name]['capacity']
        marines = []
        for marine_name, marine_info in state['marine_ships'].items():
            marine_index = marine_info['index']
            marines.append((marine_name, marine_index))
        marines = tuple(marines)
        treasures = []
        for treasure_name, treasure_info in state['treasures'].items():
            treasure_loc = treasure_info['location']
            treasures.append((treasure_name, treasure_loc))
        treasures = tuple(treasures)

        our_state = ((pirate_loc, capacity), (marines, treasures))

        return self.policy[state['turns to go']][our_state][0]

        

        



class PirateAgent:

    def __init__(self, initial):
            self.unwanted = []
            self.initial = deepcopy(initial)
            self.map = initial['map']
            self.base = get_base_loc(self.map)
            self.turns = self.initial['turns to go']
            self.pirate_names = list(self.initial['pirate_ships'].keys())
            self.threshold = 0.007 if not self.initial['optimal'] else 0
            self.start_time = time()
            self.prob_change_location = {treasure: self.initial['treasures'][treasure]['prob_change_location'] for treasure in self.initial['treasures']}
            self.initial_state = ((self.base, 2), \
                                        tuple([tuple([(name, self.initial['marine_ships'][name]['index']) for name in self.initial['marine_ships']]), \
                                        tuple([(name, self.initial['treasures'][name]['location']) for name in self.initial['treasures']])]))

            self.marine_possible_indexes = self.get_possible_movements(self.initial['marine_ships'], 'path')
            self.treasures_possible_locations = self.get_possible_movements(self.initial['treasures'], 'possible_locations')

            self.pirate_possible_locations = [(i, j) for i in range(len(self.map)) for j in range(len(self.map[0]))\
                                            if self.map[i][j] == 'S' or self.map[i][j] == 'B']
            
            self.stochastic_enviroments = self.get_all_enviroments()
            self.states = self.get_all_states()
            self.actions_from_state = self.get_actions_from_state(self.states)
            self.transition_probabilities = self.get_transition_probabilities(self.stochastic_enviroments)
            self.full_transition_probabilities = self.full_transition_probabilities()
            self.policy = self.make_policy()
            self.last_turn = 0
            
        
    
    def make_policy(self):
        policy = list()
        for i in range(self.turns + 1):
                policy.append(dict())
        for state in self.states:
            
            pirate_loc, capacity = state[0]
            marines, treasures = state[1]
            if pirate_loc in [self.initial['marine_ships'][name]['path'][idx] for (name, idx) in marines]:
                policy[0][state] = (None, -len(self.pirate_names))
            else:
                policy[0][state] = (None, 0)
            
        for turn in range(1, self.turns + 1):
            print(f"Turn: {turn}")
            for state in self.states:
                deterministic_state = state[0]
                pirate_loc, capacity = deterministic_state
                stochastic_state = state[1]
                marines, treasures = stochastic_state

                if time() - self.start_time > 60:
                    break

                if self.collision(pirate_loc, marines):
                    instant_reward = -len(self.pirate_names)
                else:
                    instant_reward = 0

                actions = self.actions_from_state[state]
                best_action = 'terminate'
                best_value = 0
                for action in actions:
                    action_reward = self.action_reward(action, state)
                    expected_value = 0
                    for next_state, probability in self.full_transition_probabilities[(state, action)].items():
                        expected_value += (probability * (policy[turn - 1][next_state][1]))
                    if action_reward + expected_value > best_value:
                        best_value = action_reward + expected_value
                        best_action = action
                policy[turn][state] = (best_action, best_value + instant_reward)
            if time() - self.start_time > 60:
                self.last_turn = turn
                break
        return policy


    def collision(self, new_pirate_loc, marines):
            for marine_name, marine_index in marines:
                if new_pirate_loc == self.initial['marine_ships'][marine_name]['path'][marine_index]:
                    return True
            return False


    def action_reward(self, action, state):
        capacity = state[0][1]
        if action == 'reset':
            return -2
        if action == 'terminate':
            return 0
        if action[0][0] == 'collect':
            return 0
        if action[0][0] == 'sail':
            return 0
        if action[0][0] == 'deposit':
            return len(self.pirate_names) * (2 - capacity) * 4
        if action[0][0] == 'wait':
            return 0
        

    def full_transition_probabilities(self):
        transitions = dict()
        for state in self.states:
            for action in self.actions_from_state[state]:
                transitions[(state, action)] = self.states_from_action(state, action)

        return transitions

    def states_from_action(self, state, action):

        states_probs = {}

        deterministic_state = state[0]
        pirate_loc, capacity = deterministic_state
        stochastic_state = state[1]

        
        for next_env, probability in self.transition_probabilities[stochastic_state].items():
            if action == 'terminate':
                return {}

            if action == 'reset':
                return {self.initial_state: 1}

            if action[0][0] == 'collect':
                new_capacity = capacity - 1
                if self.collision(pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'sail':
                new_pirate_loc = action[0][2]
                new_capacity = capacity
                if self.collision(new_pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((new_pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'deposit':
                new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability

            if action[0][0] == 'wait':
                new_capacity = capacity
                if self.collision(pirate_loc, next_env[0]):
                    new_capacity = 2
                states_probs[((pirate_loc, new_capacity), next_env)] = probability   
        return states_probs         
        


    def get_actions_from_state(self, states):
        """
        Get all possible actions from a given state

        :param states: a tuple of all possible states

        :return: a dictionary containing all possible actions from a given state
                    actions: (state: tuple, action: str)
        """

        actions_from_state = dict()

        for state in states:
            deterministic_state = state[0]
            pirate_loc, capacity = deterministic_state

            stochastic_state = state[1]
            marines, treasures = stochastic_state

            # Get all possible actions

            # terminate and reset actions
            actions = ['reset', 'terminate']

            # collect actions
            atomic = []
            for treasure_info in treasures:
                treasure_loc = treasure_info[1]
                treasure_name = treasure_info[0]

                if is_adjacent(pirate_loc, treasure_loc) and capacity > 0:
                    for pirate in self.pirate_names:
                        atomic.append(('collect', pirate, treasure_name))
                    actions.append(tuple(atomic))

            # sail actions
            
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (pirate_loc[0] + offset[0], pirate_loc[1] + offset[1]) in self.pirate_possible_locations:
                    atomic = []
                    for pirate in self.pirate_names:
                        atomic.append(('sail', pirate, (pirate_loc[0] + offset[0], pirate_loc[1] + offset[1])))
                    actions.append(tuple(atomic))

            # deposit actions
            atomic = []
            if pirate_loc == self.base and capacity < 2:
                for pirate in self.pirate_names:
                    atomic.append(('deposit', pirate))
                actions.append(tuple(atomic))

            # wait actions
            atomic = []
            for pirate in self.pirate_names:
                atomic.append(('wait', pirate))
            actions.append(tuple(atomic))

            # Generate all possible actions from the state
            actions_from_state[state] = actions

        return actions_from_state


    def get_transition_probabilities(self, stochastic_enviroments):
        
        transition = dict()
        for environment in stochastic_enviroments:
            transition[environment] = self.get_transition_probabilities_from_environment(environment)
        return transition


    def get_transition_probabilities_from_environment(self, stochastic_enviroment):

        marines, treasures = stochastic_enviroment

        marine_probs = dict()

        for marine_name, marine_index in marines:
            marine_path = self.initial['marine_ships'][marine_name]['path']

            if marine_index == len(marine_path) - 1:
                marine_next_locations_probs = {(marine_name, marine_index): 0.5, (marine_name, marine_index - 1): 0.5}
            elif marine_index == 0:
                marine_next_locations_probs = {(marine_name, marine_index): 0.5, (marine_name, marine_index + 1): 0.5}
            else:
                marine_next_locations_probs = {(marine_name, marine_index): 1/3, (marine_name, marine_index + 1): 1/3,\
                                                (marine_name, marine_index - 1): 1/3}
                
            
            marine_probs[marine_name] = marine_next_locations_probs

        treasure_probs = dict()
        for treasure_name, treasure_loc in treasures:
            treasure_next_locations_probs = dict()
            for possible_loc in self.initial['treasures'][treasure_name]['possible_locations']:
                if possible_loc == treasure_loc:
                    treasure_next_locations_probs[(treasure_name, possible_loc)] = 1 - self.prob_change_location[treasure_name]\
                                                                    + self.prob_change_location[treasure_name] / (len(self.initial['treasures'][treasure_name]['possible_locations']))  
                else:
                    treasure_next_locations_probs[(treasure_name, possible_loc)] = self.prob_change_location[treasure_name] / (len(self.initial['treasures'][treasure_name]['possible_locations']))
            
            treasure_probs[treasure_name] = treasure_next_locations_probs

        all_probs = []

        # Append marine transition probabilities
        for marine_name, transitions in marine_probs.items():
            all_probs.append([(marine_name, loc, prob) for ((marine_name, loc), prob) in transitions.items()])

        # Append treasure transition probabilities
        for treasure_name, transitions in treasure_probs.items():
            all_probs.append([(treasure_name, loc, prob) for ((treasure_name, loc), prob) in transitions.items()])

        # Dictionary to hold full transition probabilities
        full_transition_probabilities = {}

        # Compute all possible combinations of states using Cartesian product
        for combined_states in product(*all_probs):
            env = [[], []]
            p = 1
            for (name, loc, prob) in combined_states:
                p *= prob
                if type(loc) == int:
                    env[0].append((name, loc))
                else:
                    env[1].append((name, loc))
            env[0] = tuple(env[0])
            env[1] = tuple(env[1])
            if p > self.threshold:
                self.unwanted.append(env)
                full_transition_probabilities[tuple(env)] = p

        return full_transition_probabilities

        


                


    def get_all_states(self):
        """
        Get all possible states for the game

        :return: a tuple of all possible states
                    state: tuple(deterministic_state: tuple, stochastic_state: tuple)
        """

        deterministic_enviroments = tuple([(pirate_loc, capacity) for pirate_loc in self.pirate_possible_locations\
                                          for capacity in [0, 1, 2]])
        
        # Generate all combinations of deterministic_state and stochastic_state
        states = tuple(list(product(deterministic_enviroments, self.stochastic_enviroments)))
        return states


    def get_all_enviroments(self):
        """
        Get all possible enviroments for the game

        :return: a list of all possible enviroments
                    Environment: (marines_locations: tuple, treasures_locations: tuple)
                    entity_location: tuple(entity_name: str, location: tuple)
        """
        

        # Generate all combinations of marine_locations and treasure_locations
        all_marine_enviroments = list(product(*self.marine_possible_indexes))
        all_treasure_enviroments = list(product(*self.treasures_possible_locations))

        # Generate all possible enviroments
        enviroments = tuple(list(product(all_marine_enviroments, all_treasure_enviroments)))
        return enviroments


    def get_possible_movements(self, entity_info, criteria):
        """
        Get all possible movements for a given entity

        :param enity_info: a dictionary containing information about the entity
        :param criteria: the criteria to get the possible movements, either 'possible_locations' or 'path'

        :return: a tuple containing all possible entity name and the movements combinations
                    tuple(entity_name: str, possible_movements: tuple)
        """

        entity_possible_movements = list()
        for entity in entity_info:
            
            if criteria == 'possible_locations':
                entity_possible_movements.append(tuple([(entity, loc) for loc in entity_info[entity][criteria]]))
            else:
                entity_possible_movements.append(tuple([(entity, index) for index in range(len(entity_info[entity][criteria]))]))
            

        return entity_possible_movements
    

    def reward(self, state, action):
        """
        Calculate the reward of a given state and the action led to it

        :param state: the current state
        :param action: the action taken

        :return: the reward of the state
        """

        # Get the deterministic state
        deterministic_state = state[0]
        pirate_loc, capacity = deterministic_state

        # Get the stochastic state
        stochastic_state = state[1]
        marines_locations, treasures_locations = stochastic_state

        # Get the action
        if action == 'terminate':
            return 0

        


    def act(self, state):
        # make a tuple state of the state
        ship_name = list(state['pirate_ships'].keys())[0]
        pirate_loc = state['pirate_ships'][ship_name]['location']
        capacity = state['pirate_ships'][ship_name]['capacity']
        marines = []
        for marine_name, marine_info in state['marine_ships'].items():
            marine_index = marine_info['index']
            marines.append((marine_name, marine_index))
        marines = tuple(marines)
        treasures = []
        for treasure_name, treasure_info in state['treasures'].items():
            treasure_loc = treasure_info['location']
            treasures.append((treasure_name, treasure_loc))
        treasures = tuple(treasures)

        our_state = ((pirate_loc, capacity), (marines, treasures))
        print(self.policy[min(self.last_turn - 1, state['turns to go'])])
        print('State:', our_state)
        print(self.policy[min(self.last_turn - 1, state['turns to go'])])
        return self.policy[min(self.last_turn - 1, state['turns to go'])][our_state][0]
    
    


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplementedError

    def value(self, state):
        raise NotImplementedError


