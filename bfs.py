from collections import deque

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
            if 0 <= nx < rows and 0 <= ny < cols and map[nx][ny] != 'I' and distances[nx][ny] == float('inf'):
                distances[nx][ny] = current_dist + 1
                queue.append((nx, ny))

    return distances

map = [['P', 'P', 'I', 'I'],
      ['P', 'P', 'P', 'P'],
      ['I', 'P', 'I', 'P'],
      ['P', 'P', 'V', 'I']]
start_pos = (3,2)

for row in bfs(map, start_pos):
    print(row)