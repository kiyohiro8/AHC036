# 与えられたグラフを分割してquotient graphを構成し、quotient graph上の最短路を通るように目的を訪問する
import random
import networkx as nx
import numpy as np
from collections import defaultdict, deque
import random
random.seed(0)

debug = False


def partition_graph(G, P, max_nodes):
    partitions = []
    visited = [False] * len(G.nodes())
    # ノードを左下からの距離で昇順に並べたもの
    nodes = list(G.nodes())
    sorted_nodes = sorted(nodes, key=lambda x: P[x][0] ** 2 + P[x][1] ** 2)
    while len(sorted_nodes) > 0:
        # 最初のノードを取得
        node = sorted_nodes[0]
        temp_visited = visited.copy()
        temp_visited[node] = True
        stack = [node]
        partition = []
        while stack and (len(partition) < max_nodes):
            # stackを左下からの距離で昇順に並べる
            stack = sorted(stack, key=lambda x: P[x][0] ** 2 + P[x][1] ** 2, reverse=True)
            # 
            cur_node = stack.pop()
            partition.append(cur_node)
            sorted_nodes.remove(cur_node)
            visited[cur_node] = True
            for neighbor in G.neighbors(cur_node):
                if not temp_visited[neighbor]:
                    temp_visited[neighbor] = True
                    stack.append(neighbor)
            
        partitions.append(partition)
    
    return partitions

def custom_quotient_graph(G, base_partition, additional_partition):
    Q = nx.Graph()
    
    # ノードの作成 (属性1: 基本パーティション)
    for i, part in enumerate(base_partition):
        Q.add_node(i, members=list(part), type='base')
    
    # ノードの作成 (属性2: 追加パーティション)
    for i, part in enumerate(additional_partition, start=len(base_partition)):
        Q.add_node(i, members=list(part), type='additional')
    
    # エッジの作成
    all_partitions = base_partition + additional_partition
    for i, part1 in enumerate(all_partitions):
        for j, part2 in enumerate(all_partitions):
            if i < j:  # 重複を避けるため
                connecting_edges = [(n1, n2) for n1 in part1 for n2 in part2 if G.has_edge(n1, n2)]
                if connecting_edges:
                    if Q.nodes[i]['type'] == Q.nodes[j]['type'] == 'base':
                        Q.add_edge(i, j, connecting_edges=connecting_edges, edge_type='base-base')
                    elif Q.nodes[i]['type'] == Q.nodes[j]['type'] == 'additional':
                        Q.add_edge(i, j, connecting_edges=connecting_edges, edge_type='additional-additional')
                    else:
                        Q.add_edge(i, j, connecting_edges=connecting_edges, edge_type='base-additional')
    
    return Q

def extract_quotient_graph_info(Q):
    # 1. Qの各ノードの部分グラフを構成するGのノードを順番に配列Aに格納
    A = []
    # 2. Qの各ノードのタイプを示す辞書
    node_types = {}
    # 3. Aの要素iがQの何番のノードに属しているかを示す配列
    node_assignments = []
    # 4. Qのノードの何番がAの何番目から何番目の要素に対応しているのかを示す辞書
    q_node_ranges = {}
    
    current_index = 0
    for q_node, data in sorted(Q.nodes(data=True)):
        members = data['members']
        start_index = current_index
        
        A.extend(members)
        node_types[q_node] = data['type']
        node_assignments.extend([q_node] * len(members))
        
        end_index = current_index + len(members)  # 変更点：end_indexを次の要素の開始位置に
        q_node_ranges[q_node] = (start_index, end_index)
        current_index = end_index
    
    return A, node_types, node_assignments, q_node_ranges

def find_path_through_quotient(G, Q, start, end):
    node_to_q_nodes = defaultdict(list)
    for q_node, data in Q.nodes(data=True):
        for node in data['members']:
            node_to_q_nodes[node].append(q_node)
    
    # startとendのbaseノードを見つける
    start_base_nodes = [q_node for q_node in node_to_q_nodes[start] if Q.nodes[q_node]['type'] == 'base']
    end_base_nodes = [q_node for q_node in node_to_q_nodes[end] if Q.nodes[q_node]['type'] == 'base']
    
    if not start_base_nodes or not end_base_nodes:
        raise ValueError("Start or end node is not in any base partition")
    
    shortest_q_path = None
    for start_q_node in start_base_nodes:
        for end_q_node in end_base_nodes:
            try:
                q_path = nx.shortest_path(Q, start_q_node, end_q_node)
                if shortest_q_path is None or len(q_path) < len(shortest_q_path):
                    shortest_q_path = q_path
            except nx.NetworkXNoPath:
                continue
    
    if shortest_q_path is None:
        raise nx.NetworkXNoPath(f"No path between {start} and {end}")
    
    g_path = [start]
    q_nodes = [shortest_q_path[0]]
    node_direction_dict = {}
    
    for i in range(len(shortest_q_path) - 1):
        current_q_node, next_q_node = shortest_q_path[i], shortest_q_path[i+1]
        connecting_edges = Q.get_edge_data(current_q_node, next_q_node)['connecting_edges']
        #もしconnecting_edgeの中にg_path[-1]が使われているものがあるなら、そのエッジを選ぶ
        target_edge = None
        for edge in connecting_edges:
            if g_path[-1] in edge:
                target_edge = edge
                break
        if target_edge is None:
            target_edge = connecting_edges[0]
        
        current_subgraph = G.subgraph(Q.nodes[current_q_node]['members'])
        next_subgraph = G.subgraph(Q.nodes[next_q_node]['members'])

        if target_edge[0] in current_subgraph and target_edge[1] in next_subgraph:
            target_current = target_edge[0]
            target_next = target_edge[1]
        elif target_edge[0] in next_subgraph and target_edge[1] in current_subgraph:
            target_current = target_edge[1]
            target_next = target_edge[0]
        else:
            raise ValueError("Invalid target edge")
        
        if g_path[-1] == target_current:
            pass
        else:
            current_node = g_path[-1]
            current_subpath = nx.shortest_path(current_subgraph, current_node, target_current)
            g_path.extend(current_subpath[1:])
            q_nodes.extend([current_q_node] * (len(current_subpath) - 1))
        
        g_path.append(target_next)
        q_nodes.append(next_q_node)
        
    
    # 最後の部分グラフ内での移動
    last_subgraph = G.subgraph(Q.nodes[shortest_q_path[-1]]['members'])
    last_subpath = nx.shortest_path(last_subgraph, g_path[-1], end)
    g_path.extend(last_subpath[1:])
    q_nodes.extend([shortest_q_path[-1]] * (len(last_subpath) - 1))

    return g_path, q_nodes


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
    with open("./sample_input/0009.txt", "r") as f:
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

max_additional_partition_size = L_A - N
additional_partitions = []
additional_partition_size = 0

# まだ不足している場合はランダムなパスを追加
while additional_partition_size < max_additional_partition_size:
    # ランダムに決めた開始点から、ランダムに移動しながらL_B個のノードを取得
    node = random.choice(list(G.nodes()))
    visited = [False] * N
    path = [node]
    visited[node] = True
    while len(path) < L_B:
        neighbors = list(G.neighbors(node))
        next_node = random.choice(neighbors)
        if not visited[next_node]:
            path.append(next_node)
            visited[next_node] = True
        node = next_node
    
    if len(path) + additional_partition_size < max_additional_partition_size:
        additional_partitions.append(path)
        additional_partition_size += len(path)
    else:
        length = max_additional_partition_size - additional_partition_size
        additional_partitions.append(path[:length])
        additional_partition_size += length

# グラフを分割
partitions = partition_graph(G, P, L_B)
# quotient graphを構築
Q = custom_quotient_graph(G, partitions, additional_partitions)

# Aを構成
A, node_type_dict, node_assignments, q_node_range_dict = extract_quotient_graph_info(Q)
assert len(A) == L_A, f"Expected {L_A} nodes in A, but got {len(A)} nodes"
B = [-1] * L_B

final_output = []
final_output.append(" ".join(map(str, A)))

pos_from = 0
cur_node = pos_from
cur_q_node = -1

for pos_to in t:
    g_path, q_nodes = find_path_through_quotient(G, Q, cur_node, pos_to)
    g_path = g_path[1:]
    q_nodes = q_nodes[1:]

    for i in range(len(g_path)):
        g_node = g_path[i]
        if cur_node == g_node:
            continue
        # 次のノードがBに含まれているならそのまま進む
        if g_node in B:
            final_output.append(f"m {g_node}")
            cur_node = g_node
        # 次のノードがAに含まれていない場合
        else:
            # 次のノードのnode_typeがbaseかadditionalかで分岐
            node_type = node_type_dict[q_nodes[i]]
            if node_type == "base":
                start_idx, end_idx = q_node_range_dict[q_nodes[i]]
                sub_A = A[start_idx:end_idx]
                if len(sub_A) == L_B:
                    B = sub_A
                    final_output.append(f"s {len(sub_A)} {start_idx} 0")
                else:
                    assert len(sub_A) < L_B, f"Expected {L_B} nodes in sub_A, but got {len(sub_A)} nodes"


                    th = random.randint(0, L_B - len(sub_A))
                    B[th:th+len(sub_A)] = sub_A
                    final_output.append(f"s {len(sub_A)} {start_idx} {th}")
            elif node_type == "additional":
                start_idx, end_idx = q_node_range_dict[q_nodes[i]]
                sub_A = A[start_idx:end_idx]
                # 次のノードがsub_Aの何番目に存在しているかを取得
                idx = sub_A.index(g_node)
                if len(sub_A) - idx >= L_B:
                    B = sub_A[idx:idx+L_B]
                    final_output.append(f"s {L_B} {start_idx+idx} 0")
                else:
                    assert len(sub_A) - idx < L_B, f"Expected {L_B} nodes in sub_A, but got {len(sub_A) - idx} nodes"
                    th = random.randint(0, L_B - (len(sub_A) - idx))
                    B[th:th+len(sub_A)-idx] = sub_A[idx:]
                    final_output.append(f"s {len(sub_A) - idx} {start_idx+idx} {th}")

            cur_node = g_node
            final_output.append(f"m {g_node}") # 次のノードに移動

if not debug:
    for line in final_output:
        print(line)
else:
    with open("output.txt", "w") as f:
        for line in final_output:
            f.write(line + "\n")
