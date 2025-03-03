from collections import defaultdict
from utils import PriorityQueue
from itertools import product
from collections import deque, defaultdict
import heapq
from utils import NORTH, SOUTH, EAST, WEST

ids = [342663978, 337604821]

RESET_PENALTY = 2
DESTROY_HORCRUX_REWARD = 2
DEATH_EATER_PENALTY = 1

class OptimalWizardAgent:
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

        horcrux_names = list()
        death_eater_names = list()
        horcrux_locs = list()
        horcrux_probs = list()
        death_eater_indices = list()
        death_eater_locs = list()

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
                if self.map[i][j] == 'P':
                    [wizard_possible_locs[k].append((i, j)) for k in range(len(self.wizards_names))]

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
        if len(de_indices) == 1:
            return {de_index: 1}
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
                    if (tuple(hor_states), tuple(de_states)) in next_state_distribution.keys():
                        next_state_distribution[(tuple(hor_states), tuple(de_states))] += prob
                    else:
                        next_state_distribution[(tuple(hor_states), tuple(de_states))] = prob

            next_combo_dict[combo] = dict(next_state_distribution)

        return next_combo_dict
    
    def get_all_actions_from_states(self):
        actions_from_states = dict()
        for state in self.all_states:
            actions = self.get_all_actions(state)
            actions_from_states[state] = actions
        return actions_from_states

    def state_from_action(self, state, action):
        state_res = defaultdict(lambda: 0)
        if action == "reset":
            state_res = {self.initial_state: 1}
        elif action == "terminate":
            state_res = {}
        else:
            new_wizard_loc = list(state[0])
            for env in self.stochastic_environments_probs[state[1:]]:
                for i, atomic_action in enumerate(action):
                    if atomic_action[0] == "move":
                        new_wizard_loc[i] = atomic_action[2]
                    elif atomic_action[0] in ["destroy", "wait"]:
                        continue
                new_state = (tuple(new_wizard_loc), *tuple(env))
                state_res[new_state] = self.stochastic_environments_probs[state[1:]][env]
        return state_res


    def get_all_states_from_states(self):
        res = dict()
        for state in self.all_states:
            for action in self.actions_in_states[state]:
                res[state, action] = self.state_from_action(state, action)
        return res
    
    def all_possible_moves_in_tiles(self):
        offsets = NORTH, EAST, SOUTH, WEST
        move_actions = dict()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == 'P':
                    for offset in offsets:
                        new_loc = (i + offset[0], j + offset[1])
                        if 0 <= new_loc[0] < len(self.map) and \
                        0 <= new_loc[1] < len(self.map[0]) and\
                        self.map[new_loc[0]][new_loc[1]] == 'P':
                            if (i, j) not in move_actions.keys():
                                move_actions[(i, j)] = [new_loc]
                            else:
                                move_actions[(i, j)].append(new_loc)
        return move_actions


    def get_all_actions(self, state):
        wiz_locs = state[0]
        hor_locs = state[1]
        de_locs = state[2]

        def get_move_actions(wiz_locs, wiz_names):
            offsets = NORTH, EAST, SOUTH, EAST
            move_actions = [[] for _ in wiz_locs]
            for i, (loc, name) in enumerate(zip(wiz_locs, wiz_names)):
                if loc in self.moves_in_tiles.keys():
                    move_actions[i].extend([("move", name, new_loc) for new_loc in self.moves_in_tiles[loc]])
                else:
                    continue
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
        [actions[i].extend(wait_actions[i]) for i in range(len(wiz_locs))]  

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
    
    def check_collision(self, state):
        wiz_locs = state[0]
        de_inds = state[2]
        R = 0
        for wiz_loc in wiz_locs:
            for i, de_ind in enumerate(de_inds):
                de_loc = self.death_eater_locs[i][de_ind]
                if wiz_loc == de_loc:
                    R -= DEATH_EATER_PENALTY
        return R

    def append_action_to_env(self, curr_wiz_state, next_env_state, action):
        if action in ["reset", "terminate"]:
            return self.initial_state
        new_wizard_loc = list(curr_wiz_state)
        for i, atomic_action in enumerate(action):
            if atomic_action[0] == "move":
                new_wizard_loc[i] = atomic_action[2]
            elif atomic_action[0] in ["destroy", "wait"]:
                continue
        new_state = (tuple(new_wizard_loc), *tuple(next_env_state))
        return tuple(new_state)


    def value_iteration(self):
        # init virtual policy for state with no turns
        policy = [{} for _ in range(self.turns + 1)]
        for state in self.all_states:
            R = self.check_collision(state)
            policy[0][state] = ("terminate", R)
        close = False
        for k in range(1, self.turns + 1):
            for state in self.all_states:
                env_state = state[1:]
                wiz_state = state[0]
                if state not in self.actions_in_states.keys():  
                    actions = self.get_all_actions(state)
                    self.actions_in_states[state] = actions
                else:
                    actions = self.actions_in_states[state]
                # state reward AKA R(s)
                R = self.check_collision(state)
                best_action = ("terminate", 0)
                for action in actions:
                    if action == "terminate":
                        continue
                    next_envs = self.stochastic_environments_probs[env_state]
                    
                    # current action value
                    action_current_value = 0
                    if action == 'reset':
                        action_current_value -= RESET_PENALTY
                    else:
                        for atomic_action in action:
                            if atomic_action[0] == "destroy":
                                action_current_value += DESTROY_HORCRUX_REWARD

                    action_future_value = 0
                    # future action value
                    for next_env in next_envs:
                        prob = next_envs[next_env]

                        next_state = self.append_action_to_env(wiz_state, next_env, action)

                        action_future_value += prob * policy[k-1][next_state][1]
                    if (action_future_value + action_current_value) > best_action[1]:
                        best_action = (action, (action_future_value + action_current_value))
                policy[k][state] = (best_action[0], best_action[1] + R)
        return policy
    

    def __init__(self, initial):
        self.actions_in_states = dict()
        self.horcrux_names, self.horcrux_locs, self.horcrux_probs, self.death_eater_names,\
              self.death_eater_indices, self.death_eater_locs = self.extract_initial_info(initial)
        self.wizards_names = list(initial['wizards'].keys())
        self.initial_state = self.state_to_list(initial)
        self.map = initial['map']
        self.moves_in_tiles = self.all_possible_moves_in_tiles()
        self.turns = initial['turns_to_go']
        self.all_environments, self.all_states = self.generate_all_environments()
        self.stochastic_environments_probs = self.get_transition_probabilities()
        self.policy = self.value_iteration()
        
        

    def act(self, state):
        turns_to_go = state['turns_to_go']
        state_list = self.state_to_list(state)
        actions = self.policy[turns_to_go][state_list][0]
        return actions
        

class WizardAgent(OptimalWizardAgent):
    def __init__(self, initial):
        self.map = initial['map']
        self.deatheaters_info = initial['death_eaters']
        self.distances = self.get_all_distances()
        self.danger_zones = self._map_dangers()
        self.reachable = self.reachable_locations(initial)
        self.retry_count = 0

    def _is_tile_legal(self, tile):
        return 0 <= tile[0] < len(self.map) and 0 <= tile[1] < len(self.map[0]) and self.map[tile[0]][tile[1]] != 'I'
        
    def _appr_exp_hor_pos(self, best_pos, available_positions):
        closest_source = None
        min_distance = 1000

        for source in available_positions:
            if source in self.distances and best_pos in self.distances[source]:  
                dist = self.distances[source][best_pos]
                if dist < min_distance:
                    min_distance = dist
                    closest_source = source
        return closest_source
    
    def _nearest_reachable(self, pos, accessible):
        seen = set()
        q = deque()
        offsets = NORTH, SOUTH, WEST, EAST
        for offset in offsets:
            neighbor = (pos[0] + offset[0], pos[1] + offset[1])
            if not self._is_tile_legal(neighbor):
                continue
            q.append((neighbor, 1))
            seen.add(neighbor)
        while q:
            cell, d = q.popleft()
            if cell in self.reachable and cell in accessible:
                return cell
            for offset in offsets:
                neighbor = (pos[0] + offset[0], pos[1] + offset[1])
                if not self._is_tile_legal(neighbor):
                    continue
                if neighbor not in seen:
                    seen.add(neighbor)
                    q.append((neighbor, d+1))
    
    def _exp_hor_pos(self, poss_list, cur_pos):
        eps = 0.00001
        total_w = 0
        sum_x = 0
        sum_y = 0
        accessible_candidates = [p for p in poss_list if p in self.reachable]
        if not accessible_candidates:
            return cur_pos
        for cand in accessible_candidates:
            d = self.island_case(cur_pos, cand)
            weight = 1 / (d + eps)
            sum_x += cand[0] * weight
            sum_y += cand[1] * weight
            total_w += weight
        avg_x = sum_x / total_w
        avg_y = sum_y / total_w
        best = (round(avg_x), round(avg_y))
        if best in self.reachable:
            return best
        else:
            return self._nearest_accessible(best, accessible_candidates)
    
    def _map_dangers(self):
        risk_levels = defaultdict(int)
        for foe in self.deatheaters_info.values():
            prob = 0
            if len(foe['path']) == 2:
                prob = 1.5
            elif len(foe['path']) == 1:
                prob = 3
            else:
                prob = 1
            for position in foe["path"]:
                risk_levels[position] += prob
        return risk_levels

    def bfs(self, loc):
        q = deque([loc])
        bfs_tree = dict()
        bfs_tree[loc] = 0
        offsets = NORTH, SOUTH, EAST, WEST
        seen = set([loc])
        while q:
            curr_tile = q.popleft()
            curr_distance = bfs_tree[curr_tile]
            for (dx, dy) in offsets:
                next_tile = (curr_tile[0] + dx, curr_tile[1] + dy)
                if not self._is_tile_legal(next_tile):
                    continue
                if self._is_tile_legal(next_tile) and\
                   next_tile not in bfs_tree:
                    
                    bfs_tree[next_tile] = curr_distance + 1
                    q.append(next_tile)
                    seen.add(next_tile)
        return bfs_tree
    
    def get_all_distances(self):
        distances = dict()
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] != 'I':
                    distances[(i,j)] = self.bfs((i,j))
        return distances
    
    def reachable_locations(self, state):
        reachable = set()
        offsets = NORTH, SOUTH, EAST, WEST
        for wiz_name, wiz_data in state['wizards'].items():
            wiz_loc = wiz_data['location']
            q = deque([wiz_loc])
            seen = set()
            seen.add(wiz_loc)
            while q:
                curr_node = q.popleft()
                reachable.add(curr_node)
                for offset in offsets:
                    next_node = (curr_node[0] + offset[0], curr_node[1] + offset[1])
                    if not self._is_tile_legal(next_node):
                        continue
                    if next_node not in seen:
                        seen.add(next_node)
                        q.append(next_node)
        return reachable
    
    
    def find_path(self, start, goal):
        open_list = [(0, start)]
        came_from = {}
        g = {start: 0}
        f = {start: self.island_case(start, goal)}
        offsets = NORTH, SOUTH, WEST, EAST
        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1] 
            for offset in offsets:
                neighbor = (current[0] + offset[0], current[1] + offset[1])
                if not self._is_tile_legal(neighbor):
                    continue
                tmp = g[current] + 1 + self.danger_zones.get(neighbor, 0)
                if tmp < g.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g[neighbor] = tmp
                    f[neighbor] = tmp + self.island_case(neighbor, goal)
                    heapq.heappush(open_list, (f[neighbor], neighbor))
        return []
    
    def island_case(self, a, b):
        if self.map[a[0]][a[1]] != 'I':
            if b not in self.distances[a]:
                return 1000
            else:
                return self.distances[a][b]
        else:
            return 1000

    def act(self, state):
        wizards = state['wizards']
        horcruxes = state['horcrux']
        actions = []
        horcrux_locations = {name: info["location"] for name, info in horcruxes.items()}
        optimal_destinations = {name: self._exp_hor_pos(info['possible_locations'], info['location']) for name, info in horcruxes.items()}

        for wiz_name, wiz_data in wizards.items():
            pos = wiz_data["location"]
            best_target = min(optimal_destinations.items(), key=lambda x: self.island_case(pos,x[1]))
            horcrux_name, destination = best_target

            if pos == destination and pos in horcrux_locations.values():
                actions.append(("destroy", wiz_name, horcrux_name))
                continue
            
            step = self.find_path(pos, destination)
            if step:
                actions.append(("move", wiz_name, step[0]))
            else:
                alternative = self._backup_movement(pos, destination)
                if alternative:
                    actions.append(("move", wiz_name, alternative))
                else:
                    actions.append(("wait", wiz_name))
                    self.retry_count += 1
        if not actions:
            if self.retry_count > 3:
                return "reset"
        
        return tuple(actions)

    def _optimal_horcrux_spot(self, horcrux):
        possible_sites = horcrux["possible_locations"]
        current = horcrux["location"]
        return min(possible_sites, key=lambda site: self.distances[current][site]) if possible_sites else current

    def _backup_movement(self, pos, goal):
        potential_moves = []
        offsets = [(1,0),(0,1),(-1,0),(0,-1)]
        for offset in offsets:
            if self._is_tile_legal((pos[0] + offset[0], pos[1] + offset[1])):
                potential_moves.append((pos[0] + offset[0], pos[1] + offset[1]))
        return min(potential_moves, key=lambda step: (self.island_case(step,goal) if potential_moves else None))

