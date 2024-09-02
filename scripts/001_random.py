# Aをランダムに生成
# 信号操作と移動を次のように行う
# 1. 現在地と目的地の最短路をBFSで求める
# 2. 1で求めた経路を現在の信号状態が青であるところまで移動する。目的地に到達したなら1に戻る
# 3. 配列Aの中で、path上の次の場所にあたるノードのインデックスを探す
# 4. 3で見つけたインデックスを含む長さPBの部分配列をランダムに選ぶ
# 5. 4で選んだ部分配列でBを上書き
# 6. 2に戻る

from collections import deque
import time
import random
random.seed(0)

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

def generate_A(n, l_a):
    # 大きさl_aの配列を生成。各要素は0以上n-1以下。ただし、0からn-1の各要素が少なくとも1つ含まれる。
    A = list(range(n)) + [random.randint(0, n-1) for _ in range(l_a - n)]
    random.shuffle(A)
    return A


# get input
N, M, T, L_A, L_B = map(int, input().split())

G = [[] for _ in range(N)]

for _ in range(M):
    u, v = map(int, input().split())
    G[u].append(v)
    G[v].append(u)

t = list(map(int, input().split()))

P = []
for _ in range(N):
    x, y = map(int, input().split())
    P.append((x, y))

time_start = time.time()
# 最小値を求めたいので大きな値で初期化
best_count = 10 ** 6
start = time.time()
while time.time() - time_start < 2.5:
    final_output = []
    signal_change_count = 0
    # construct and output the array A
    A = generate_A(N, L_A)
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
                B = A[pb_start:pb_end]
                signal_change_count += 1
                final_output.append(f"s {pb_end - pb_start} {pb_start} 0")
            if cur == pos_to:
                pos_from = pos_to
                break
        

    if signal_change_count < best_count:
        best_count = signal_change_count
        best_output = final_output

for line in best_output:
    print(line)



