
# import ex2_dpll as ex2
# import ex2_gilad as ex2
# import ex2_also as ex2
import ex2
import inputs_gil as inputs
from copy import deepcopy
import time
import os

CODES_NEW = {'passage': 0, 'dragon': 1, 'vault': 2, 'trap': 3, 'hollow_vault': 4, 'vault_trap': 5, 'dragon_trap': 6,
             'hollow_trap_vault': 7}
INVERSE_CODES_NEW = dict([(j, i) for i, j in CODES_NEW.items()])
ACTION_TIMEOUT = 5
CONSTRUCTOR_TIMEOUT = 60
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
INFINITY = 100000


class Checker:
    def __init__(self):
        pass

    def check_controller(self):
        pass

    def true_state_to_controller_input(self):
        pass

    def is_action_legal(self, action):
        pass

    def change_state_after_action(self, action):
        pass

    def at_goal(self):
        pass


class GringottsChecker(Checker):
    game_map: list
    harry_cur_loc: tuple
    turn_limit: int
    dragon_locs: list
    trap_locs: list
    vault_locs: list
    hollow_loc: tuple
    collected_hollow: bool


    def __init__(self, input):
        super().__init__()
        self.path = []
        self.game_map = input['full_map']
        self.harry_cur_loc = input['Harry_start']
        self.dragon_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                            if 'dragon' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.trap_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                          if 'trap' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.vault_locs = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                           if 'vault' in INVERSE_CODES_NEW[self.game_map[x][y]]]
        self.hollow_loc = [(x, y) for x in range(len(self.game_map)) for y in range(len(self.game_map[x]))
                           if 'hollow' in INVERSE_CODES_NEW[self.game_map[x][y]]][0]  # Should only be one
        m = len(self.game_map)
        n = len(self.game_map[0])
        self.turn_limit = 5 + 3 * (n + m)
        self.collected_hollow = False
        # print(f"Maximal amount of turns is {self.turn_limit}!")

    def check_controller(self):
        map_dimensions = (len(self.game_map), len(self.game_map[0]))
        observations = self.create_observations()
        constructor_start = time.time()
        controller = ex2.GringottsController(map_dimensions, self.harry_cur_loc, observations)
        constructor_finish = time.time()
        if constructor_finish - constructor_start > CONSTRUCTOR_TIMEOUT:
            return f"Timeout on constructor! Took {constructor_finish - constructor_start} seconds," \
                   f" should take no more than {CONSTRUCTOR_TIMEOUT}"

        counter = 0
        while not self.at_goal():
            observations = self.create_observations()
            start = time.time()
            action = controller.get_next_action(observations)
            finish = time.time()
            # if finish - start > ACTION_TIMEOUT:
            #     return -2
                # return f"Timeout on action! Took {finish - start} seconds, should take no more than {ACTION_TIMEOUT}"
            if not self.is_action_legal(action):
                return -1
                # return f"Action {action} is illegal! Either because the action is impossible or because Harry dies"
            counter += 1
            if counter > self.turn_limit:
                return 0
                # return "Turn limit exceeded!"
            self.change_state_after_action(action)
        # return f"{counter} < {self.turn_limit}"
        return f"{counter}"
        # return f"Goal achieved in {counter} steps!"

    def create_state(self):
        return self.harry_cur_loc

    def create_observations(self):
        observations = []
        close_locs = self.get_close_locs()
        for loc in close_locs:
            if 'dragon' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(('dragon', loc))
            if 'vault' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(('vault', loc))
            if 'trap' in INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]:
                observations.append(tuple(['sulfur']))
        observations = list(set(observations))  # Remove duplicates of traps
        return observations

    def is_action_legal(self, action):
        if len(action) == 0 or len(action) >= 3:
            return False
        if len(action) == 1:
            if action[0] == 'wait':
                return True
            if action[0] == 'collect':
                return True
            else:
                return False
        else:
            close_locs = self.get_close_locs()
            if action[0] == 'destroy':
                if action[1] in close_locs:
                    return True
                else:
                    return False
            elif action[0] == 'move':
                new_loc = action[1]
                if new_loc in close_locs:
                    if new_loc in self.dragon_locs:
                        return False
                    if new_loc in self.trap_locs:
                        return False
                    return True
            return False

    def get_close_locs(self):
        harry_y, harry_x = self.harry_cur_loc
        num_rows = len(self.game_map)
        num_cols = len(self.game_map[0])
        return [(harry_y + direction[0], harry_x + direction[1]) for direction in DIRECTIONS
                if ((0 <= harry_y + direction[0] < num_rows) and
                    (0 <= harry_x + direction[1] < num_cols))]

    def change_state_after_action(self, action):
        if action[0] == "move":
            self.path.append(action)
            self.change_state_after_moving(action)
        elif action[0] == "destroy":
            self.path.append(action)
            self.change_state_after_destroy(action)
        elif action[0] == "collect":
            self.path.append(action)
            self.change_state_after_collect()

    def change_state_after_moving(self, action):
        _, loc = action
        self.harry_cur_loc = loc

    def change_state_after_destroy(self, action):
        _, loc = action
        if loc in self.trap_locs:
            self.trap_locs.remove(loc)
            prev_status = INVERSE_CODES_NEW[self.game_map[loc[0]][loc[1]]]
            if prev_status == 'trap':
                new_status = 'passage'
            elif prev_status == 'vault_trap':
                new_status = 'vault'
            elif prev_status == 'dragon_trap':
                new_status = 'dragon'
            else:
                new_status = 'hollow_vault'
            self.game_map[loc[0]][loc[1]] = CODES_NEW[new_status]

    def change_state_after_collect(self):
        if self.harry_cur_loc == self.hollow_loc:
            self.collected_hollow = True

    def at_goal(self):
        return self.collected_hollow



def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_symbol(code):
    symbols = {
        0: '⬜',  # passage
        1: '🐉',  # dragon
        2: '🏰',  # vault
        3: '⚡',  # trap
        4:  '🎁' ,  # hollow vault
        5: '💥',  # vault trap
        6: '🔥',  # dragon trap
        7: '⚜️',  # hollow trap vault
    }
    return symbols.get(code, '❓')


def print_board(board, path=None):
    current_path = set(path) if path else set()

    for i in range(len(board)):
        row = ""
        for j in range(len(board[i])):
            if (i, j) in current_path:
                row += "\033[42m" + get_symbol(board[i][j]) + "\033[0m"  # Green background for entire path
            else:
                row += get_symbol(board[i][j])
        print(row)
    print("\n")


def animate_path(board, actions):
    path = []

    # Convert actions from the given format to a list of tuples
    processed_actions = [(action[0], tuple(action[1]) if len(action) > 1 else None)
                         for action in actions]

    clear_screen()
    print("Initial board:")
    print_board(board)
    time.sleep(2)

    for action, coords in processed_actions:
        clear_screen()
        if action == 'move':
            path.append(coords)
        print(f"Action: {action}")
        if coords:
            print(f"Position: {coords}")
        print_board(board, path)
        time.sleep(0.5)

if __name__ == '__main__':


    def rotate_90(grid):
        """Rotate the grid 90 degrees clockwise."""
        return [list(row) for row in zip(*grid[::-1])]


    def generate_rotations(grid):
        """Generate all 6 unique rotations of the grid."""
        rotations = [grid]  # Start with the original grid
        for _ in range(3):  # Generate 90°, 180°, 270° rotations
            grid = deepcopy(rotate_90(grid))
            rotations.append(grid)
        transposed = deepcopy([list(row) for row in zip(*rotations[0])])  # Transpose of the original grid
        rotations.append(transposed)  # Include the non-rotated transpose
        for _ in range(3):  # Generate 90°, 180°, 270° rotations of the transpose
            transposed = rotate_90(transposed)
            rotations.append(transposed)
        return rotations


    def update_loading_bar(progress, total, bar_length=40):
        """Update and display a loading bar based on progress."""
        percent = progress / total if total > 0 else 1
        filled_length = int(bar_length * percent)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        print(f"\r|{bar}| {progress}/{total} ({percent * 100:.2f}%)", end="")
        if progress == total:
            print()


    def calculate_total_tests(level):
        """Calculate the total number of tests for a given level."""
        total_tests = 0
        for input in level:
            grid = input['full_map']
            # Count passages (0s) in the original grid
            zeros = sum(1 for row in grid for item in row if item == 0)
            # Multiply by 8 since we test all 8 possible orientations (4 rotations + 4 flipped rotations)
            total_tests += zeros * 8
        return total_tests


    def check_board(input, i, flag=False, flag2=True):
        """Perform checks on a board, updating progress and counters."""
        if flag2:
            first = True
            try:
                grid = input['full_map']
                rotations = generate_rotations(grid)
                if flag:
                    print(GringottsChecker(input).check_controller())
                for rotation in rotations:
                    # Collect valid locations for the current rotation
                    locs = [tuple([r, c]) for r in range(len(rotation)) for c in range(len(rotation[0])) if
                            rotation[r][c] == 0]

                    # Test each location on the current rotation
                    for loc in locs:
                        test_input = deepcopy(input)
                        test_input['Harry_start'] = loc
                        test_input['full_map'] = deepcopy(rotation)
                        my_checker = GringottsChecker(test_input)
                        total[i] += 1
                        result = int(my_checker.check_controller())
                        if result > 0:
                            cnt[i] += 1
                        elif first:
                            GringottsChecker(test_input).check_controller()
                            first = False
            except Exception:
                pass
        else:
            checker = GringottsChecker(input)
            print(checker.check_controller())


    if __name__ == '__main__':
        # Initialize counters
        cnt = [0, 0, 0, 0]  # TA examples, lv1, lv2, lv3
        total = [0, 0, 0, 0]
        flag = False
        f = False
        choose_seeds = {69: True, 42: True, 31415926: True, 960210: True, 'TA': True}
        levels = [[], [], []]

        # Prepare levels based on seeds
        if choose_seeds[69]:
            levels[0].extend(inputs.inputlv1_69)
            levels[1].extend(inputs.inputlv2_69)
            levels[2].extend(inputs.inputlv3_69)
        if choose_seeds[42]:
            levels[0].extend(inputs.inputlv1_42)
            levels[1].extend(inputs.inputlv2_42)
            levels[2].extend(inputs.inputlv3_42)
        if choose_seeds[960210]:
            levels[0].extend(inputs.inputlv1_960210)
            levels[1].extend(inputs.inputlv2_960210)
            levels[2].extend(inputs.inputlv3_960210)
        if choose_seeds[31415926]:
            levels[0].extend(inputs.inputlv1_31415926)
            levels[1].extend(inputs.inputlv2_31415926)
            levels[2].extend(inputs.inputlv3_31415926)

        # Calculate total tests for each level
        total_tests_per_level = [calculate_total_tests(level) for level in levels]
        current_progress = [0, 0, 0]  # Track progress for each level

        print("\n----------------TA tests:----------------\n")
        if choose_seeds['TA']:
            for input in inputs.inputs:
                check_board(input, 0)

        # Iterate through levels
        for i, (level, total_tests) in enumerate(zip(levels, total_tests_per_level)):
            if len(level) > 0:
                print(f"\n----------------level {i + 1} tests:----------------\n")
                for input in level:
                    check_board(input, i + 1, flag)
                    # Update progress based on the actual number of tests performed
                    zeros_in_board = sum(1 for row in input['full_map'] for item in row if item == 0)
                    current_progress[i] += zeros_in_board * 8  # 8 orientations per position
                    update_loading_bar(current_progress[i], total_tests)

        # Print results
        results = [(c, t, round(c / t, 3)) for c, t in zip(cnt, total) if t > 0]
        print("\npassed, total number, Percent of boards passed")
        print(results)
        print(
            f"Overall stats, solved {sum(cnt)} boards out of {sum(total)} with an accuracy of: {round(sum(cnt) / sum(total), 3)} ")