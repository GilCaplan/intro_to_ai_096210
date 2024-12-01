import problems

map = [['P', 'P', 'I', 'I'],
      ['P', 'P', 'P', 'P'],
      ['I', 'P', 'I', 'P'],
      ['P', 'P', 'V', 'I']]
start_pos = (0,0)
coords = [(1,2), (2,1)]

def dfs(map, start, impassable_coords, target_value='V'):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = len(map), len(map[0])
    stack = [start]
    visited = set()
    while stack:
        x, y = stack.pop()
        if map[x][y] == 'I' or (x, y) in impassable_coords:
            # print(f"can't pass here ({x},{y}) ")
            continue

        if map[x][y] == target_value:
            return True
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                stack.append((nx, ny))
    return False
print(dfs(map, start_pos, coords))