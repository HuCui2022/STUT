import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]
        self.hops, self.dis_adjs = self.get_distance_graph()


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_distance_graph(self):
        A = self.A.sum(0)  # shape: 25, 25, adjmatrix
        A[A != 0] = 1
        dis_adj = [None for _ in range(25)]
        dis_adj[0] = np.eye(25)
        dis_adj[1] = A
        hops = 0 * dis_adj[0]  # values ==0 , shape : 25 25
        for i in range(2, 25):
            dis_adj[i] = dis_adj[i - 1] @ A.transpose(0, 1)
            dis_adj[i][dis_adj[i] != 0] = 1  # 得到第 i 阶的矩阵

        for i in range(25 - 1, 0, -1):
            if np.any(dis_adj[i] - dis_adj[i - 1]):
                dis_adj[i] = dis_adj[i] - dis_adj[i - 1]  # 去掉重复的部分，也就是只在 i 阶出现，但是不在 i-1阶出现的值。
                hops += i * dis_adj[i]
            else:
                dis_adj[i] *= 0.0
                # 这里不该是continue
                # self.hops : 表示不同整数的矩阵。表示点和点之间的相对距离。

        # self.hops = torch.tensor(self.hops).long()  # shape : 25,25,25 -> 25,25 将不同相对距离矩阵进行合并。
        # self.hops shape : 25 25 values is differnte distance
        dis_adjs = np.concatenate([np.expand_dims(A, 0) for A in dis_adj], 0)

        return hops, dis_adjs


# g = Graph()
# # print(g.hops.shape)
# # print(len(g.dis_adjs))
# # print(g.dis_adjs[12])
# print(g.hops.max())   # 11