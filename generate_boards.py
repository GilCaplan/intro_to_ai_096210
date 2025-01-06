import random

CODES_NEW = {
    'passage': 0,
    'dragon': 1,
    'vault': 2,
    'trap': 3,
    'hollow_vault': 4,
    'vault_trap': 5,
    'dragon_trap': 6,
    'hollow_trap_vault': 7
}
random.seed(69)

def generate_board(rows, cols, num_dragons, num_vaults, num_traps, deathly_hallow=True):
    board = [[CODES_NEW['passage'] for _ in range(cols)] for _ in range(rows)]
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]

    if deathly_hallow:
        deathly_hallow_type = random.choices([CODES_NEW['hollow_vault'], CODES_NEW['hollow_trap_vault']], [0.75, 0.25])[
            0]
        deathly_hallow_position = random.choice(all_positions)
        r, c = deathly_hallow_position
        board[r][c] = deathly_hallow_type
        all_positions.remove(deathly_hallow_position)

    total_tiles = rows * cols
    max_items = int(total_tiles * 0.75)

    dragon_positions = random.sample(all_positions, min(num_dragons, len(all_positions))) if num_dragons > 0 else []
    for r, c in dragon_positions:
        board[r][c] = CODES_NEW['dragon']
        all_positions.remove((r, c))
    max_items -= len(dragon_positions)

    vault_positions = random.sample(all_positions, min(num_vaults, len(all_positions)))
    for r, c in vault_positions:
        board[r][c] = random.choice([CODES_NEW['vault'], CODES_NEW['vault_trap']])
        all_positions.remove((r, c))
    max_items -= len(vault_positions)

    trap_positions = random.sample(all_positions, min(num_traps, len(all_positions)))
    for r, c in trap_positions:
        board[r][c] = CODES_NEW['trap']
        all_positions.remove((r, c))
    max_items -= len(trap_positions)

    def is_solvable():
        from collections import deque

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = deque([(0, 0)])
        reachable_positions = 0

        while queue:
            current_pos = queue.popleft()
            if current_pos in visited:
                continue

            visited.add(current_pos)
            x, y = current_pos
            if board[x][y] == CODES_NEW['dragon']:
                continue
            reachable_positions += 1
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols and
                        (nx, ny) not in visited):
                    queue.append((nx, ny))
        return reachable_positions > 1

    attempts = 0
    max_attempts = 100
    while not is_solvable() and attempts < max_attempts:
        board = [[CODES_NEW['passage'] for _ in range(cols)] for _ in range(rows)]
        all_positions = [(r, c) for r in range(rows) for c in range(cols)]

        if deathly_hallow:
            deathly_hallow_position = random.choice(all_positions)
            r, c = deathly_hallow_position
            board[r][c] = deathly_hallow_type
            all_positions.remove(deathly_hallow_position)

        dragon_positions = random.sample(all_positions, min(num_dragons, len(all_positions))) if num_dragons > 0 else []
        for r, c in dragon_positions:
            board[r][c] = CODES_NEW['dragon']
            all_positions.remove((r, c))

        vault_positions = random.sample(all_positions, min(num_vaults, len(all_positions)))
        for r, c in vault_positions:
            board[r][c] = random.choice([CODES_NEW['vault'], CODES_NEW['vault_trap']])
            all_positions.remove((r, c))

        trap_positions = random.sample(all_positions, min(num_traps, len(all_positions)))
        for r, c in trap_positions:
            board[r][c] = CODES_NEW['trap']
            all_positions.remove((r, c))

        attempts += 1

    if attempts >= max_attempts:
        raise ValueError("Could not generate a solvable board after maximum attempts")

    if board[0][0] == CODES_NEW['passage']:
        loc = (0, 0)
    else:
        loc = None
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == CODES_NEW['passage']:
                    loc = (r, c)
                    break
            if loc is not None:
                break
        if loc is None:
            raise ValueError("No valid starting position found")

    return board, loc


def generate_examples(num_boards, difficulty_level):
    examples = []
    for _ in range(num_boards):
        rows = random.randint(4, 6)
        cols = random.randint(4, 6)

        if difficulty_level == 1:
            num_dragons = 1
            num_vaults = 2
            num_traps = 3
            deathly_hallow = True
        elif difficulty_level == 2:
            num_dragons = random.randint(1, 4)
            num_vaults = 3
            num_traps = 4
            deathly_hallow = True
        elif difficulty_level == 3:
            num_dragons = random.randint(1, 4)
            num_vaults = random.randint(3, 5)
            num_traps = random.randint(4, 6)
            deathly_hallow = True

        try:
            board, start_loc = generate_board(rows, cols, num_dragons, num_vaults, num_traps, deathly_hallow)
            examples.append({
                'Harry_start': start_loc,
                'full_map': board
            })
        except ValueError as e:
            print(f"Skipping board generation due to error: {e}")
            continue

    return examples

level_1_boards = generate_examples(10, 1)
level_2_boards = generate_examples(5, 2)
level_3_boards = generate_examples(5, 3)

def print_boards(boards, level):
    for idx, board in enumerate(boards):
        print("{")
        print(f"    'Level': {level},")
        print(f"    'Board': {idx + 1},")
        print(f"    'Harry_start': {board['Harry_start']},")
        print("    'full_map': [")
        for row in board['full_map']:
            print(f"        {row},")
        print("    ]")
        print("},\n")


print("inputlv1 = [")
print_boards(level_1_boards, 1)
print("]\ninputlv2 = [")
print_boards(level_2_boards, 2)
print("]\ninputlv3 = [")
print_boards(level_3_boards, 3)
print("]")
