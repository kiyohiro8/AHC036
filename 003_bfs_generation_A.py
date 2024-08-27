# Aの生成手順を変更
# 0から開始して、BFSで見つけたノードを順番にAに追加していく。Aのサイズが余ったらランダムなノードを追加する。

from collections import deque
import time
import networkx as nx
import random
random.seed(0)

debug = True

def bfs(cur, target, graph):
    queue = deque([[cur]])
    visited = [False] * len(graph)
    visited[cur] = True
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == target:
            return path
        
        for neighbor in graph[node]:
            if visited[neighbor] == False:
                visited[neighbor] = True
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

def generate_A(graph, l_a):
    # ノード0から開始して、BFSで見つけたノードを順番にAに追加していく
    # このグラフは連結であることが保証されている。
    A = []
    visited = [False] * len(graph)
    queue = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        A.append(node)
        for neighbor in graph[node]:
            if visited[neighbor] == False:
                visited[neighbor] = True
                queue.append(neighbor)

    # Aのサイズが余ったらランダムなノードを追加する
    while len(A) < l_a:
        node = random.randint(0, len(graph) - 1)
        A.append(node)

    return A   


# get input
if not debug:
    N, M, T, L_A, L_B = map(int, input().split())

    nodes = list(range(N))
    edges = [list(map(int, input().split())) for _ in range(M)]

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    t = list(map(int, input().split()))

    P = []
    for _ in range(N):
        x, y = map(int, input().split())
        P.append((x, y))
else:
    with open("./sample_input/0001.txt", "r") as f:
        input_lines = [line.strip() for line in f.readlines()]
    N, M, T, L_A, L_B = map(int, input_lines.pop(0).split())

    G = nx.Graph()

    for _ in range(M):
        u, v = map(int, input_lines.pop(0).split())
        G.add_edge(u, v)

    t = list(map(int, input_lines.pop(0).split()))

    P = []
    for _ in range(N):
        x, y = map(int, input_lines.pop(0).split())
        P.append((x, y))

time_start = time.time()
# 最小値を求めたいので大きな値で初期化
best_count = 10 ** 6
start = time.time()
while time.time() - time_start < 2.5:
    final_output = []
    signal_change_count = 0
    # construct and output the array A
    A = generate_A(G, L_A)
    B = [-1] * L_B
    final_output.append(" ".join(map(str, A)))

    pos_from = 0
    cur = pos_from

    for pos_to in t:

        # determine the path by DFS
        path = bfs(pos_from, pos_to, G)


        path = path[1:]

        # 2. 1で求めた経路を現在の信号状態が青であるところまで移動する。目的地に到達したなら1に戻る
        i = 0
        while i < len(path):
            next_node = path[i]
            if next_node in B:
                cur = next_node
                final_output.append(f"m {next_node}")
                i += 1
            else:
                # 3. 配列Aの中で、path上の次の場所にあたるノードのインデックスを探す
                next_index = A.index(next_node)
                # 4. 3で見つけたインデックスを含む長さPBの部分配列をランダムに選ぶ
                left_min = max(0, next_index - L_B + 1)
                left_max = min(L_A - L_B, next_index)
                pb_start = random.randint(left_min, left_max)
                pb_end = pb_start + L_B
                B = A[pb_start:pb_end].copy()
                signal_change_count += 1
                final_output.append(f"s {pb_end - pb_start} {pb_start} 0")
            if cur == pos_to:
                pos_from = pos_to
                break 

    if signal_change_count < best_count:
        best_count = signal_change_count
        best_output = final_output
if not debug:
    for line in best_output:
        print(line)
else:
    with open("output.txt", "w") as f:
        for line in best_output:
            f.write(line + "\n")


