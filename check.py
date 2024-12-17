import ex1
import search
import time
from problems import comp_problems, check_problems, non_comp_problems, s_problems
from problemsT import t_problems, t_hard_problems

def timeout_exec(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            # try:
            self.result = func(*args, **kwargs)
            # except Exception as e:
            #    self.result = (-3, -3, e)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        return default
    else:
        return it.result


def check_problem(p, search_method, timeout):
    """ Constructs a problem using ex1.create_wumpus_problem,
    and solves it using the given search_method with the given timeout.
    Returns a tuple of (solution length, solution time, solution)"""

    """ (-2, -2, None) means there was a timeout
    (-3, -3, ERR) means there was some error ERR during search """

    t1 = time.time()
    s = timeout_exec(search_method, args=[p], timeout_duration=timeout)
    t2 = time.time()


    if isinstance(s, search.Node):
        solve = s
        solution = list(map(lambda n: n.action, solve.path()))[1:]
        return (len(solution), t2 - t1, solution)
    elif s is None:
        return (-2, -2, None)
    else:
        return s


def solve_problems(problems):
    solved = 0
    cnt = 0
    for problem in problems:
        try:
            p = ex1.create_harrypotter_problem(problem)
        except Exception as e:
            print("Error creating problem: ", e)
            return None
        timeout = 60
        result = check_problem(
            p, (lambda p: search.astar_search(p, p.h)), timeout)
        print(f"{result[0]} {result[1]}, {result[2]}")
        # print(f"{result[0]} {result[1]}")
        cnt += 1
        if result[2] != None:
            if result[0] != -3:
                solved = solved + 1

        # visualize_solution(problem, result[2], use_ANSI=False)
    return

def visualize_solution(init_state, solution, use_ANSI=False):
    symbols = {
        'P': '⬜',
        'I': '🟫',
        'V': '🟥'
    }

    wizard_symbol = '🧙'
    death_eater_symbol = '👻'
    horcrux_symbol = '🔥'
    dead_symbol = '💀'
    harry_symbol = '🧙🏿'

    map = init_state['map']
    vx, vy = -1, -1

    # Finding voldemort
    for i in range(len(map)):
        for j in range(len(map[0])):
            if map[i][j] == 'V':
                vx, vy = i, j

    # Preparing state
    horcruxes = {}
    for i, horcrux in enumerate(init_state['horcruxes']):
        if tuple(horcrux) not in horcruxes:
            horcruxes[tuple(horcrux)] = [i]
        else:
            horcruxes[tuple(horcrux)].append(i)

    wizards = init_state['wizards']
    death_eaters = {name: [path, True, 0] for name, path in init_state['death_eaters'].items()}
    def print_state():
        map_copy = [row[:] for row in map]

        for name, (path, right, index) in death_eaters.items():
            x, y = path[index]
            map_copy[x][y] = death_eater_symbol  # Death Eater symbol

        for (x, y), _ in horcruxes.items():
            map_copy[x][y] = horcrux_symbol  # Horcrux symbol

        for wizard, (position, _) in wizards.items():
            x, y = position
            map_copy[x][y] = wizard_symbol

            if wizard == 'Harry Potter':
                map_copy[x][y] = harry_symbol

        if symbols['V'] == dead_symbol:
            map_copy[vx][vy] = dead_symbol

        if use_ANSI:
            print(f"\033[{len(map)}A", end="")
        else:
            print('\n' * len(map))

        for row in map_copy:
            print(''.join(symbols.get(cell, cell) for cell in row))

    print_state()
    # print(solution)

    # Updating state
    for action in solution:
        for atomic_action in action:
            action_name, details = atomic_action[0], atomic_action[1:]

            if action_name == 'move':
                wizard_name, (x, y) = details
                _, lives = wizards[wizard_name]
                wizards[wizard_name] = ((x, y), lives)
            elif action_name == 'destroy':
                wizard_name, i = details
                (x, y), lives = wizards[wizard_name]

                horcruxes[(x, y)].remove(i)
                if not horcruxes[(x, y)]:
                    horcruxes.pop((x, y))
            elif action_name == 'kill':
                symbols['V'] = dead_symbol

        for name, (path, right, index) in death_eaters.items():
            if len(path) != 1:
                if right:
                    index += 1

                    if index == (len(path)-1):
                        right = False
                else:
                    index -= 1

                    if index == 0:
                        right = True

            death_eaters[name] = (path, right, index)

            x, y = path[index]

            # I DO NOT LIKE THIS
            for wizard_name, ((wx, wy), lives) in list(wizards.items()):
                if (wx, wy) == (x, y):
                    lives -= 1

                    if lives == 0:
                        raise Exception('DANGER WTF POPPED A WIZARD . . .')
                    else:
                        wizards[wizard_name] = ((x, y), lives)

        print(action)
        time.sleep(0.5)
        print_state()


def main():
    print(ex1.ids)
    """Here goes the input you want to check"""
    print("Solving Non Complex Problems:")
    solve_problems(non_comp_problems)
    print("Solving Complex Problems:")
    solve_problems(comp_problems)
    # solve_problems(s_problems)
    # print("Solving Tal Problems:")
    # solve_problems(t_hard_problems)
    # print("Solving Check Problems:")
    # solve_problems(check_problems)
    print("done")


if __name__ == '__main__':
    main()
