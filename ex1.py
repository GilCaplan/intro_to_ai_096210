import search
import random
import math
import itertools
import json
from copy import deepcopy


ids = ["111111111", "111111111"]


class HarryPotterProblem(search.Problem):

    """This class implements a medical problem according to problem description file"""
    def __init__(self, initial):
        def update_death_eaters_path(death_eaters):
            def compute_path(death_eater):
                path = death_eater + list(reversed(death_eater[1:len(death_eater) - 1]))
                return path

            death_eater_paths = {}
            for de in death_eaters.keys():
                death_eater_paths[de] = compute_path(death_eaters[de])
            return death_eater_paths
        self.map = initial['map']
        self.move_num = 0
        self.voldemort_killed = False

        initial_state = {
            'wizards': initial['wizards'],
            'death_eaters': update_death_eaters_path(initial['death_eaters']),
            'horcruxes': initial['horcruxes'],
        }
        self.voldemort_loc = (0, 0)
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 'V':
                    self.voldemort_loc = (i, j)
                    break
        initial_state = json.dumps(initial_state)
        search.Problem.__init__(self, initial_state)

    def actions(self, state):
        """Return the valid actions that can be executed in the given state."""
        state = json.loads(state)
        wizards = state['wizards']
        horcruxes = state['horcruxes']

        def get_move_actions(loc, wiz_name):
            move_actions = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                if 0 <= new_loc[0] < len(self.map) and 0 <= new_loc[1] < len(self.map[0])\
                    and self.map[new_loc[0]][new_loc[1]] != 'I':
                    move_actions.append(('move', wiz_name, new_loc))
            return tuple(move_actions)
        
        def get_destroy_horcrux_actions(loc, wiz_name):
            destroy_actions = []
            for horcrux_loc in horcruxes:
                if loc == horcrux_loc:
                    destroy_actions.append(('destroy', wiz_name, horcrux_loc))
            return tuple(destroy_actions)
        
        def get_wait_actions(wiz_name):
            wait = [('wait', wiz_name)]
            return tuple(wait)
        
        def get_kill_voldemort_action(loc, wiz_name):
            if wiz_name == 'Harry Potter' and self.map[loc[0]][loc[1]] == 'V' and len(horcruxes) == 0: 
                 #TODO: maybe check if theres been at least one turn since the last horcrux has been destroyed
                return tuple([('kill', wiz_name)])
            return ()
        
        actions = []
        for wizard in wizards:
            actions.append(get_move_actions(wizards[wizard][0], wizard) + get_destroy_horcrux_actions(wizards[wizard][0], wizard)\
                         + get_wait_actions(wizard) + get_kill_voldemort_action(wizards[wizard][0], wizard))
        actions = tuple(itertools.product(*actions))   #TODO: check if this is the correct way to get all possible actions
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given action in the given state."""
        state = json.loads(state)
        new_state = deepcopy(state)
        wizards = new_state['wizards']
        self.move_num += 1
        for atomic_action in action:
            action_name = atomic_action[0]
            wiz_name = atomic_action[1]
            new_loc = state['wizards'][wiz_name][0]
            if action_name == 'move':
                new_loc = atomic_action[2]
                wizards[wiz_name] = (new_loc, wizards[wiz_name][1])
            
            elif action_name == 'wait':
                continue

            elif action_name == 'destroy':
                new_state['horcruxes'].remove(atomic_action[2])

            elif action_name == 'kill':
                self.voldemort_killed = True

            for de in new_state['death_eaters']:
                curr_loc = self.move_num % len(new_state['death_eaters'][de])
                if tuple(new_state['death_eaters'][de][curr_loc]) == tuple(new_loc):
                    new_state['wizards'][wiz_name] = (new_loc, wizards[wiz_name][1] - 1)

        return json.dumps(new_state)
    

    def goal_test(self, state):
        """Return True if the state is a goal state."""
        return self.voldemort_killed
    

    def h(self, node):
        """
        Heuristic function for A* search.
        Estimates the minimum number of moves needed to reach the goal.
        """

        def manhattan_distance(loc1, loc2):
            return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

        state = json.loads(node.state)
        wizards = state['wizards']
        death_eater_paths = state['death_eaters']
        horcruxes = state['horcruxes']

        if horcruxes:
            distances = []
            wizard_names = list(wizards.keys())

            for wizard in wizard_names:
                wizard_loc = wizards[wizard][0]
                distances.append([manhattan_distance(wizard_loc, h) for h in horcruxes])

            min_cost = float('inf')
            for permutation in itertools.permutations(range(len(horcruxes)), len(wizard_names)):
                total_cost = sum(distances[w][permutation[w]] for w in range(len(wizard_names)) if w < len(horcruxes))
                min_cost = min(min_cost, total_cost)

            horcrux_cost = min_cost
        else:
            horcrux_cost = 0

        death_eater_penalty = 0
        for wizard_name in wizards:
            wizard_loc = wizards[wizard_name][0]

            for death_eater, path in death_eater_paths.items():
                next_position = path[(self.move_num + 1) % len(path)]
                for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_loc = (wizard_loc[0] + offset[0], wizard_loc[1] + offset[1])
                    if new_loc == next_position:
                        death_eater_penalty += 10  # Apply penalty for being in the same spot as a Death Eater

        harry_loc = wizards.get("Harry Potter", [(0, 0)])[0]
        voldemort_cost = manhattan_distance(harry_loc, self.voldemort_loc) if not horcruxes else 0
        # return 1
        return horcrux_cost + voldemort_cost + death_eater_penalty

        # IDEAS:   1. distance from harry to voldemort + #horcruxes
        #          2. distance from harry to voldemort + min_distance from the wizards to the horcruxes
    


def create_harrypotter_problem(game):
    return HarryPotterProblem(game)
