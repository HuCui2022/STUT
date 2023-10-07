import sys
import numpy as np
#  pick up with one hand,
#  pick up with two hands,
#  drop trash,
#  walk around,
#  sit down,
#  stand up,
#  donning,
#  doffing,
#  throw,
#  carry.
sys.path.extend(['../'])
from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
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
        dis_adj = [None for _ in range(20)]
        dis_adj[0] = np.eye(20)
        dis_adj[1] = A
        hops = 0 * dis_adj[0]  # values ==0 , shape : 25 25
        for i in range(2, 20):
            dis_adj[i] = dis_adj[i - 1] @ A.transpose(0, 1)
            dis_adj[i][dis_adj[i] != 0] = 1  # 得到第 i 阶的矩阵

        for i in range(20 - 1, 0, -1):
            if np.any(dis_adj[i] - dis_adj[i - 1]):
                dis_adj[i] = dis_adj[i] - dis_adj[i - 1]  # 去掉重复的部分，也就是只在 i 阶出现，但是不在 i-1阶出现的值。
                hops += i * dis_adj[i]
            else:
                dis_adj[i] *= 0.0
                # 这里不该是continue
                # self.hops : 表示不同整数的矩阵。表示点和点之间的相对距离。

        # self.hops = torch.tensor(self.hops).long()  # shape : 20 20 将不同相对距离矩阵进行合并。
        # self.hops shape : 20 20 values is differnte distance
        dis_adjs = np.concatenate([np.expand_dims(A, 0) for A in dis_adj], 0)
        return hops, dis_adjs

'''g = Graph()
# print(g.hops.shape)
# print(len(g.dis_adjs))
print(g.dis_adjs[11])
# print(g.hops.max())   # 10'''