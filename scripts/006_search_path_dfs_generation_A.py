# 004のgenerate_AをDFSに変更

from collections import deque
import time
import networkx as nx
import random
random.seed(0)

debug = False



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
    # ノード0から開始して、DFSで見つけたノードを順番にAに追加していく
    # このグラフは連結であることが保証されている。
    A = []
    visited = [False] * len(graph)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        A.append(node)
        for neighbor in graph[node]:
            if visited[neighbor] == False:
                visited[neighbor] = True
                stack.append(neighbor)

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

final_output = []
signal_change_count = 0
# construct and output the array A
A = generate_A(G, L_A)
B = [-1] * L_B
final_output.append(" ".join(map(str, A)))

pos_from = 0
cur = pos_from

for pos_to in t:

    while cur != pos_to:

        # 現在のノードから青信号ノードを通って到達可能なノードを探す
        queue = deque([[cur]])
        visited = [False] * N
        visited[cur] = True
        reachable_nodes_path = [[cur]]
        # bfsで到達可能なノードを探す
        while queue:
            path = queue.popleft()
            node = path[-1]
            # 次の目的地に到達可能なのであれば探索を打ち切る
            if node == pos_to:
                next_path = path
                break
            # まだ訪れていない隣接ノードの中で青信号のノードをキューに加える
            for neighbor in G[node]:
                if visited[neighbor] == False and neighbor in B:
                    visited[neighbor] = True
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
                    reachable_nodes_path.append(new_path)
        reachable_nodes = [p[-1] for p in reachable_nodes_path]  
            

        if pos_to in reachable_nodes:
            path = [p for p in reachable_nodes_path if p[-1] == pos_to][0]
            # 目的地に到達可能なので、青信号ノードを通って目的地に到達するpathを求める
            # 青信号ノードを通って目的地に到達するpathを求める
            for node in path[1:]:
                final_output.append(f"m {node}")
                cur = node
        # 次の目的地までは到達可能でない場合到達可能ノードの中で目的地に最も近いノードを選択
        else:
            queue = deque([pos_to])
            visited = [False] * N
            visited[pos_to] = True

            while queue:
                node = queue.popleft()
                if node in reachable_nodes:
                    next_node = node
                    break
                for neighbor in G[node]:
                    if visited[neighbor] == False:
                        visited[neighbor] = True
                        queue.append(neighbor)
            next_path = [p for p in reachable_nodes_path if p[-1] == next_node][0]

            # next_pathの最終地点が現在位置でない場合
            if next_path[-1] != cur:
                for node in next_path[1:]:
                    final_output.append(f"m {node}")
                    cur = node
            # next_pathの最終地点が現在位置の場合は何もしない
            else:
                pass

            # 現在位置が到達可能ノードの中で次の目的地に最も近い場合、Bの内容を変更する
            # まず現在位置から次の目的地までの最短路を求める
            path = bfs(cur, pos_to, G)
            indices = [i for i, x in enumerate(A) if x == path[1]]
            section_list = [(i, i+ 1) for i in indices if i + 1 <= len(A)]
            node_index = 1
            temp_list = A[section_list[0][0]:section_list[0][1]]
            while len(temp_list) < L_B:
                node_index += 1
                if node_index >= len(path):
                    break
                #まず最短路の次のノードがAのどこにあるかのインデックスを取得
                indices = [i for i, x in enumerate(A) if x == path[node_index]]
                new_section_list = []
                for section in section_list:
                    start, end = section
                    for i in indices:
                        if start <= i and i < end:
                            new_section_list.append(section)
                            break
                        elif i < start:
                            if end - i <= L_B:
                                new_section_list.append((i, end))
                                break
                        elif i >= end:
                            if i - start <= L_B:
                                new_section_list.append((start, i))
                                break
                if new_section_list:
                    section_list = new_section_list
                else:
                    break
            section = random.choice(section_list)
            temp_list = A[section[0]:section[1]]
            if L_B - len(temp_list) > 0:
                th = random.randint(0, L_B - len(temp_list))
            elif L_B - len(temp_list) == 0:
                th = 0
            else:
                print(L_B - len(temp_list))
                raise ValueError("L_B - len(temp_list) must be equal or greater than 0")
            B[th:th + len(temp_list)] = temp_list
            final_output.append(f"s {len(temp_list)} {section[0]} {th}")

if not debug:
    for line in final_output:
        print(line)
else:
    with open("output.txt", "w") as f:
        for line in final_output:
            f.write(line + "\n")


