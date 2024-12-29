import random

# Constants for map elements
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

# Helper function to generate a random board
# Helper function to generate a random board
def generate_board(rows, cols, num_dragons, num_vaults, num_traps, deathly_hallow=True):
    # Initialize the board with all passages (0)
    board = [[CODES_NEW['passage'] for _ in range(cols)] for _ in range(rows)]

    # Prepare a list of all positions (row, col) on the board
    all_positions = [(r, c) for r in range(rows) for c in range(cols)]

    # Step 1: Ensure Deathly Hallow (either hollow_vault or hollow_trap_vault)
    if deathly_hallow:
        deathly_hallow_type = random.choices([CODES_NEW['hollow_vault'], CODES_NEW['hollow_trap_vault']], [0.75, 0.25])[0]
        deathly_hallow_position = random.choice(all_positions)
        board[deathly_hallow_position[0]][deathly_hallow_position[1]] = deathly_hallow_type
        all_positions.remove(deathly_hallow_position)  # Remove position from available set

    # Step 2: Calculate maximum number of items (75% of the board)
    total_tiles = rows * cols
    max_items = int(total_tiles * 0.75)  # Maximum number of items allowed

    # Step 3: Place Dragons
    dragon_positions = random.sample(all_positions, num_dragons) if num_dragons > 0 else []
    for pos in dragon_positions:
        board[pos[0]][pos[1]] = CODES_NEW['dragon']
        all_positions.remove(pos)  # Remove position from available set

    # Update max_items after placing dragons
    max_items -= num_dragons

    # Step 4: Place Vaults (1 to 4, one of them is the Deathly Hallows)
    vault_positions = random.sample(all_positions, num_vaults)
    for pos in vault_positions:
        # 50% chance to place a trapped vault
        board[pos[0]][pos[1]] = random.choice([CODES_NEW['vault'], CODES_NEW['vault_trap']])
        all_positions.remove(pos)  # Remove position from available set

    # Update max_items after placing vaults
    max_items -= num_vaults

    # Step 5: Place Traps (1 to 4)
    trap_positions = random.sample(all_positions, num_traps)
    for pos in trap_positions:
        board[pos[0]][pos[1]] = CODES_NEW['trap']
        all_positions.remove(pos)  # Remove position from available set

    # Update max_items after placing traps
    max_items -= num_traps

    # Step 6: Ensure the board is solvable by ensuring there is at least one valid path
    def is_solvable():
        # Use a BFS (Breadth-First Search) to check if there's a path from (0, 0) to anywhere else
        from collections import deque

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        visited = set()
        queue = deque([(0, 0)])

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # If it's a dragon, skip this position (since dragons are impassable)
            if board[x][y] == CODES_NEW['dragon']:
                continue

            # Enqueue the neighboring cells if they are within bounds and not visited
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    # Allow moving through passage, vault, trap, or hollow (trap or vault types)
                    if board[nx][ny] != CODES_NEW['dragon']:  # Ignore dragons (impassable)
                        queue.append((nx, ny))

        # Return True if we've visited enough cells to consider it solvable
        return len(visited) > 1  # We want at least (0, 0) and one more cell visited

    # If the board is not solvable, restart the generation process
    while not is_solvable():
        board = generate_board(rows, cols, num_dragons, num_vaults, num_traps, deathly_hallow)

    return board


# Function to generate a set of boards based on difficulty level
def generate_examples(num_boards, difficulty_level):
    examples = []
    for _ in range(num_boards):
        # Randomly select board size between 3x3 to 6x6 for level 1, 4x4 to 6x6 for level 2 and 3
        rows = random.randint(4, 6) if difficulty_level == 1 else random.randint(4, 6)
        cols = random.randint(4, 6) if difficulty_level == 1 else random.randint(4, 6)

        # Difficulty Level 1: No dragons, 3 vaults (1 Deathly Hallow), 4 traps
        if difficulty_level == 1:
            num_dragons = 1
            num_vaults = 2
            num_traps = 3
            deathly_hallow = True

        # Difficulty Level 2: 1 to 4 dragons, 3 vaults (1 Deathly Hallow), 4 traps
        elif difficulty_level == 2:
            num_dragons = random.randint(1, 4)
            num_vaults = 3
            num_traps = 4
            deathly_hallow = True

        # Difficulty Level 3: 1 to 4 dragons, 3 to 5 vaults (1 Deathly Hallow), 4 to 6 traps
        elif difficulty_level == 3:
            num_dragons = random.randint(1, 4)
            num_vaults = random.randint(3, 5)
            num_traps = random.randint(4, 6)
            deathly_hallow = True

        # Generate the board with the given size and elements
        board = generate_board(rows, cols, num_dragons, num_vaults, num_traps, deathly_hallow)

        # Add the generated board to the examples list
        examples.append({
            'Harry_start': (0, 0),
            'full_map': board
        })

    return examples

# Generate 5 new boards for each difficulty level
level_1_boards = generate_examples(10, 1)
level_2_boards = generate_examples(5, 2)
level_3_boards = generate_examples(5, 3)

# Print the generated boards in the desired format for each difficulty level
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

# Print the boards for each difficulty level
print_boards(level_1_boards, 1)
# print_boards(level_2_boards, 2)
# print_boards(level_3_boards, 3)
