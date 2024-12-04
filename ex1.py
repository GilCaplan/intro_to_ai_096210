import itertools
import json
import sys
from collections import deque
from functools import lru_cache
import search

ids = ["111111111", "111111111"]


class HarryPotterProblem(search.Problem):
    def __init__(self, initial):
        def parse_map_to_integers(map):
            parsed_map = []
            for row in map:
                parsed_row = []
                for cell in row:
                    if cell == 'I':
                        parsed_row.append(0)
                    elif cell == 'P':
                        parsed_row.append(1)
                    elif cell == 'V':
                        parsed_row.append(2)
                    else:
                        raise ValueError(f"Unexpected map cell value: {cell}")
                parsed_map.append(parsed_row)
            return parsed_map

        def bfs(map, start):
            rows, cols = len(map), len(map[0])
            distances = [[float('inf')] * cols for _ in range(rows)]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            queue = deque([start])
            distances[start[0]][start[1]] = 0

            while queue:
                x, y = queue.popleft()
                current_dist = distances[x][y]

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and distances[nx][ny] == float('inf') \
                            and map[nx][ny] != 'I':
                        distances[nx][ny] = current_dist + 1
                        queue.append((nx, ny))
            return distances

        def update_death_eaters_path(death_eaters):
            def compute_path(death_eater):
                path = death_eater + list(reversed(death_eater[1:len(death_eater) - 1]))
                return path

            death_eater_paths = {}
            for de in death_eaters.keys():
                death_eater_paths[de] = compute_path(death_eaters[de])
            return death_eater_paths

        self.map = parse_map_to_integers(initial['map'])
        self.death_eaters = update_death_eaters_path(initial['death_eaters'])
        initial_state = {
            'wizards': initial['wizards'],
            'horcruxes': {i: [hor, False] for i, hor in enumerate(initial['horcruxes'])},
            'move_num': 0,
            'horcruxes_destroyed': sys.maxsize,
            'voldemort_killed': False,
        }
        self.low_num_horcruxes = len(initial_state['horcruxes']) < 4
        self.small_board = len(self.map) * len(self.map[0]) < 21
        self.voldemort_loc = next(
            ((i, j) for i, row in enumerate(self.map) for j, tile in enumerate(row) if tile == 2), None)

        self.shortest_dist_from_voldemort = bfs(self.map, self.voldemort_loc)
        initial_state = json.dumps(initial_state)
        search.Problem.__init__(self, initial_state)

    @lru_cache(maxsize=None)
    def compute_min_manhattan_distance(self, wizard_loc, horcrux_positions):
        return min(abs(wizard_loc[0] - hx) + abs(wizard_loc[1] - hy) for hx, hy in horcrux_positions)

    @lru_cache(maxsize=None)
    def compute_max_manhattan_distance(self, wizard_loc, horcrux_positions):
        return max(abs(wizard_loc[0] - hx) + abs(wizard_loc[1] - hy) for hx, hy in horcrux_positions)

    @lru_cache(maxsize=None)
    def compute_min_distance(self, horcrux_positions):
        return min([self.shortest_dist_from_voldemort[x][y] for x, y in horcrux_positions])

    @lru_cache(maxsize=None)
    def compute_max_distance_voldermort(self, horcrux_positions):
        if self.small_board and self.low_num_horcruxes:
            return 0
        return max([self.shortest_dist_from_voldemort[x][y] for x, y in horcrux_positions]) + 1

    def actions(self, state):
        """Return the valid actions that can be executed in the given state."""
        new_state = json.loads(state)
        wizards = new_state['wizards']
        horcruxes = new_state['horcruxes']

        def get_move_actions(loc, wiz_name):
            move_actions = []
            for offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_loc = (loc[0] + offset[0], loc[1] + offset[1])
                if 0 <= new_loc[0] < len(self.map) and 0 <= new_loc[1] < len(self.map[0]) \
                        and self.map[new_loc[0]][new_loc[1]] != 0:
                    if self.map[new_loc[0]][new_loc[1]] == 2:
                        if wiz_name != 'Harry Potter':
                            continue
                        if wiz_name == 'Harry Potter' and not all(horcrux[1] for horcrux in horcruxes.values()):
                            continue
                    move_actions.append(('move', wiz_name, new_loc))
            return tuple(move_actions)

        def get_destroy_horcrux_actions(loc, wiz_name):
            destroy_actions = []
            for horcrux in horcruxes.keys():
                if loc == horcruxes[horcrux][0] and not horcruxes[horcrux][1]:
                    destroy_actions.append(('destroy', wiz_name, horcrux))
            return tuple(destroy_actions)

        def get_wait_actions(wiz_name):
            return tuple([('wait', wiz_name)])

        def get_kill_voldemort_action(loc, wiz_name):
            # Check if all horcruxes are destroyed
            all_horcruxes_destroyed = all(horcrux[1] for horcrux in horcruxes.values())

            if wiz_name == 'Harry Potter' and self.map[loc[0]][loc[1]] == 2 and all_horcruxes_destroyed:
                if new_state['move_num'] > new_state['horcruxes_destroyed']:
                    return tuple([('kill', wiz_name)])
            return ()

        remaining_horcruxes = sum(1 for horcrux in horcruxes.values() if not horcrux[1])
        actions = []
        for wizard in wizards:
            wiz_loc = wizards[wizard][0]
            is_safe = not any([wiz_loc == p[new_state['move_num'] % len(p)] \
                               for p in self.death_eaters.values()])
            destroy_hocrox = get_destroy_horcrux_actions(wiz_loc, wizard)
            if len(destroy_hocrox) > 0 and is_safe:
                actions.append(destroy_hocrox)
            elif remaining_horcruxes == 0 and is_safe:
                if wizard == 'Harry Potter':
                    if self.voldemort_loc == wiz_loc:
                        get_kill_voldemort_action(wiz_loc, wizard)
                        continue
                    actions.append(
                        get_move_actions(wiz_loc, wizard, ) +
                        get_kill_voldemort_action(wiz_loc, wizard) +
                        get_wait_actions(wizard)
                    )
                else:
                    actions.append(get_wait_actions(wizard))
            else:
                actions.append(
                    get_move_actions(wiz_loc, wizard, ) +
                    destroy_hocrox +
                    get_wait_actions(wizard) +
                    get_kill_voldemort_action(wiz_loc, wizard))
        actions = tuple(itertools.product(*actions))
        return actions

    def result(self, state, action):
        """Return the state that results from executing the given action in the given state."""
        # try avoid moves where wizard dies from death eater?
        new_state = json.loads(state)
        wizards = new_state['wizards']
        horcruxes = new_state['horcruxes']
        new_state['move_num'] = new_state['move_num'] + 1
        for atomic_action in action:
            action_name = atomic_action[0]
            wiz_name = atomic_action[1]
            new_loc = new_state['wizards'][wiz_name][0]
            if action_name == 'move':
                new_loc = atomic_action[2]
                wizards[wiz_name] = (new_loc, wizards[wiz_name][1])

            elif action_name == 'wait':
                continue

            elif action_name == 'destroy':
                new_state['horcruxes'][atomic_action[2]][1] = True
                if sum(1 for horcrux in horcruxes.values() if not horcrux[1]) == 0:
                    new_state['horcruxes_destroyed'] = new_state['move_num']

            elif action_name == 'kill':
                new_state['voldemort_killed'] = True

            for de in self.death_eaters.keys():
                curr_loc = new_state['move_num'] % len(self.death_eaters[de])
                if tuple(self.death_eaters[de][curr_loc]) == tuple(new_loc):
                    new_state['wizards'][wiz_name] = (new_loc, wizards[wiz_name][1] - 1)
        return json.dumps(new_state)

    def goal_test(self, state):
        """Return True if the state is a goal state."""
        return json.loads(state)['voldemort_killed']

    def h(self, node):
        """
        Heuristic function for A* search.
        Estimates the minimum number of moves needed to reach the goal.
        """
        new_state = json.loads(node.state)
        wizards = new_state['wizards']
        horcruxes = new_state['horcruxes']
        # adding to score if wizard dies since it's GAME OVER in this case
        cost = any(100 for wiz in wizards if wizards[wiz][1] <= 0)

        # deal with destroying horcruxes
        remaining_horcruxes = sum(1 for horcrux in horcruxes.values() if not horcrux[1])
        horcrux_dist = 0
        if remaining_horcruxes > 0:
            horcrux_positions = tuple([(x, y) for [(x, y), _] in horcruxes.values()])
            cost += self.compute_max_distance_voldermort(horcrux_positions)
            for wizard in wizards.keys():
                wizard_loc = tuple(wizards[wizard][0])
                horcrux_dist += self.compute_min_manhattan_distance(wizard_loc, horcrux_positions)

            cost += remaining_horcruxes + horcrux_dist

        # Harry searching for Voldemort
        # using pre-computed distances from voldermort through BFS
        else:
            new_state['horcruxes_destroyed'] = min([new_state['move_num'], new_state['horcruxes_destroyed']])
            x, y = new_state['wizards']['Harry Potter'][0]
            cost += self.shortest_dist_from_voldemort[x][y]

            if abs(wizards['Harry Potter'][0][0] - self.voldemort_loc[0]) + abs(
                    wizards['Harry Potter'][0][1] - self.voldemort_loc[1]) > 0:
                cost += 1 if new_state['voldemort_killed'] else 0
        return cost


def create_harrypotter_problem(game):
    return HarryPotterProblem(game)
